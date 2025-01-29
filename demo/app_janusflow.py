import gradio as gr
import torch
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
from diffusers.models import AutoencoderKL
import numpy as np

# 设置设备选择逻辑
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = 'mps'
    dtype = torch.float32  # MPS设备使用float32
else:
    device = 'cpu'
    dtype = torch.float32

# Load model and processor
model_path = "deepseek-ai/JanusFlow-1.3B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = MultiModalityCausalLM.from_pretrained(model_path)
vl_gpt = vl_gpt.to(dtype).to(device).eval()

# remember to use bfloat16 dtype, this vae doesn't work with fp16
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae = vae.to(dtype).to(device).eval()

# Multimodal Understanding function
@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device, dtype=dtype)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    return answer


@torch.inference_mode()
def generate(
    input_ids,
    cfg_weight: float = 2.0,
    num_inference_steps: int = 30
):
    # we generate 5 images at a time, *2 for CFG
    tokens = torch.stack([input_ids] * 10).to(device)
    tokens[5:, 1:] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    print(inputs_embeds.shape)

    # we remove the last <bog> token and replace it with t_emb later
    inputs_embeds = inputs_embeds[:, :-1, :] 
    
    # generate with rectified flow ode
    # step 1: encode with vision_gen_enc
    z = torch.randn((5, 4, 48, 48), dtype=dtype).to(device)
    
    dt = 1.0 / num_inference_steps
    dt = torch.zeros_like(z).to(device).to(dtype) + dt
    
    # step 2: run ode
    attention_mask = torch.ones((10, inputs_embeds.shape[1]+577)).to(device)
    attention_mask[5:, 1:inputs_embeds.shape[1]] = 0
    attention_mask = attention_mask.int()
    for step in range(num_inference_steps):
        # prepare inputs for the llm
        z_input = torch.cat([z, z], dim=0) # for cfg
        t = step / num_inference_steps * 1000.
        t = torch.tensor([t] * z_input.shape[0]).to(dt)
        z_enc = vl_gpt.vision_gen_enc_model(z_input, t)
        z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
        z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
        z_emb = vl_gpt.vision_gen_enc_aligner(z_emb)
        llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

        # input to the llm
        # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
        if step == 0:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                             use_cache=True, 
                                             attention_mask=attention_mask)
            past_key_values = []
            for kv_cache in past_key_values:
                k, v = kv_cache[0], kv_cache[1]
                past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
            past_key_values = tuple(past_key_values)
        else:
            outputs = vl_gpt.language_model.model(inputs_embeds=llm_emb, 
                                             use_cache=True, 
                                             attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # transform hidden_states back to v
        hidden_states = vl_gpt.vision_gen_dec_aligner(vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
        hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
        v = vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
        v_cond, v_uncond = torch.chunk(v, 2)
        v = cfg_weight * v_cond - (cfg_weight-1.) * v_uncond
        z = z + dt * v
        
    # step 3: decode with vision_gen_dec and sdxl vae
    decoded_image = vae.decode(z / vae.config.scaling_factor).sample
    
    images = decoded_image.float().clip_(-1., 1.).permute(0,2,3,1).cpu().numpy()
    images = ((images+1) / 2. * 255).astype(np.uint8)
    
    return images
    
def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   num_inference_steps=30):
    # Clear CUDA cache if using CUDA device
    if device == 'cuda':
        torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        messages = [{'role': 'User', 'content': prompt},
                    {'role': 'Assistant', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.LongTensor(tokenizer.encode(text))
        images = generate(input_ids,
                                   cfg_weight=guidance,
                                   num_inference_steps=num_inference_steps)
        return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(images.shape[0])]

        

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    # with gr.Row():
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
        
    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")

    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
            [
                "explain this meme",
                "./images/doge.png",
            ],
            [
                "Convert the formula into latex code.",
                "./images/equation.png",
            ],
        ],
        inputs=[question_input, image_input],
    )
    
        
    gr.Markdown(value="# Text-to-Image Generation")

    
    
    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=2, step=0.5, label="CFG Weight")
        step_input = gr.Slider(minimum=1, maximum=50, value=30, step=1, label="Number of Inference Steps")

    prompt_input = gr.Textbox(label="Prompt")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    examples_t2i = gr.Examples(
        label="Text to image generation examples.",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
            "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
        ],
        inputs=prompt_input,
    )
    
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, step_input],
        outputs=image_output
    )

demo.launch(share=True)