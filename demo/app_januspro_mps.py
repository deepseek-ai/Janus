"""
app_januspro_v4.py

An updated version of your Janus Pro demo script forcing float16 on MPS,
ensuring the main model and the vision submodule share the same dtype.
"""

import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
import time

# 1. Detect device (cuda vs. mps vs. cpu)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 2. Choose dtype
#    - We'll use bfloat16 on CUDA if you prefer that, but you can use float16 if desired.
#    - We force float16 on MPS to avoid any mismatch. CPU -> float32 fallback.
if device == "cuda":
    dtype = torch.bfloat16  # or torch.float16 if you want half on CUDA
elif device == "mps":
    dtype = torch.float16   # definitely float16 on Apple MPS to avoid mismatch
else:
    dtype = torch.float32

print(f"Using device = {device}, dtype = {dtype}")

# 3. Load model config & model
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)

# If needed, force some config changes:
language_config = config.language_config
language_config._attn_implementation = 'eager'

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    language_config=language_config,
    trust_remote_code=True
)

# 4. Move entire model to the chosen device & dtype
vl_gpt = vl_gpt.to(device, dtype=dtype)

# 4a. Explicitly recast the vision submodule in case it didn't propagate
#     This helps if the vision submodel is loaded or stored differently.
if hasattr(vl_gpt, "gen_vision_model"):
    vl_gpt.gen_vision_model = vl_gpt.gen_vision_model.to(device, dtype=dtype)

# Debug prints: just to confirm
print(">>> Top-level param dtype:", next(vl_gpt.parameters()).dtype)
print(">>> Vision model param dtype:",
      next(vl_gpt.gen_vision_model.parameters()).dtype if hasattr(vl_gpt, "gen_vision_model") else "N/A")

# 5. Load processor
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 6. Utility to clear device cache (no-op for MPS/CPU)
def clear_device_cache():
    if device == "cuda":
        torch.cuda.empty_cache()

# 7. Unified seed setting
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

# 8. Multimodal Understanding
@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    clear_device_cache()
    set_seed(int(seed))

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = [Image.fromarray(image)]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(device=device, dtype=dtype)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=(False if temperature == 0 else True),
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# 9. Low-level image generation logic
def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 2,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    clear_device_cache()

    tokens = torch.zeros(
        (parallel_size * 2, len(input_ids)),
        dtype=torch.int,
        device=device
    )
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image),
        dtype=torch.int,
        device=device
    )

    pkv = None

    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv
            )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = vl_gpt.gen_head(hidden_states[:, -1, :])

            # Conditioned vs. Unconditioned
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            # Classifier-free guidance
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            # Sample
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Next token also goes to uncond
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
                dim=1
            ).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            # Force the correct dtype if needed
            if img_embeds.dtype != dtype:
                img_embeds = img_embeds.to(dtype)

            inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // patch_size, height // patch_size]
    )

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img

# 10. Text-to-Image Generation
@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    clear_device_cache()

    if seed is not None:
        set_seed(int(seed))

    width = 384
    height = 384
    parallel_size = 2

    with torch.no_grad():
        messages = [
            {'role': '<|User|>', 'content': prompt},
            {'role': '<|Assistant|>', 'content': ''}
        ]

        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag

        input_ids = torch.LongTensor(tokenizer.encode(text)).to(device)

        output, patches = generate(
            input_ids,
            width=(width // 16 * 16),
            height=(height // 16 * 16),
            cfg_weight=guidance,
            parallel_size=parallel_size,
            temperature=t2i_temperature
        )

        images = unpack(
            patches,
            width=(width // 16 * 16),
            height=(height // 16 * 16),
            parallel_size=parallel_size
        )

        pil_images = [
            Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS)
            for i in range(parallel_size)
        ]
        return pil_images

# 11. Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
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
                "images/doge.png",
            ],
            [
                "Convert the formula into latex code.",
                "images/equation.png",
            ],
        ],
        inputs=[question_input, image_input],
    )

    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (More detail => better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")
    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    examples_t2i = gr.Examples(
        label="Text to image generation examples.",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "The face of a beautiful girl",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A glass of red wine on a reflective surface.",
            "A cute and adorable baby fox with big brown eyes...",
            "The image features an intricately designed eye set against a circular backdrop...",
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
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

demo.launch(share=True)
