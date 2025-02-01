import gradio as gr
import torch
import inspect
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_processor(model_path):
    """Load and configure the VLChatProcessor with proper parameter filtering"""
    # Get valid initialization parameters
    init_params = inspect.getfullargspec(VLChatProcessor.__init__).args
    init_params.remove('self')
    
    # Load model config to find processor parameters
    model_config = AutoConfig.from_pretrained(model_path)
    processor_config = getattr(model_config, 'processor_config', {})
    
    # Filter valid parameters
    valid_config = {k: v for k, v in processor_config.items() if k in init_params}
    
    return VLChatProcessor.from_pretrained(
        model_path,
        **valid_config,
        legacy=False,
        use_fast=True
    )

def load_model():
    """Load the model with proper configuration and device management"""
    model_path = "deepseek-ai/Janus-Pro-7B"
    
    # Load model config
    config = AutoConfig.from_pretrained(model_path)
    config.language_config._attn_implementation = 'eager' if device.type == 'cpu' else 'flash_attention_2'
    
    # Load model with mixed precision
    torch_dtype = torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type != 'cpu' else None
    )
    
    
    # Load processor and tokenizer
    vl_chat_processor = load_processor(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    if device.type == 'cuda':
        vl_gpt = vl_gpt.to(device)
    
    return vl_gpt, vl_chat_processor, tokenizer

try:
    vl_gpt, vl_chat_processor, tokenizer = load_model()
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

@torch.inference_mode()
def multimodal_understanding(image, question, seed=42, top_p=0.95, temperature=0.1, max_new_tokens=1024):
    """Handle multimodal understanding requests"""
    try:
        # Input processing
        conversation = [{
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image]
        }, {"role": "<|Assistant|>", "content": ""}]
        
        # Process images and text
        pil_images = [Image.fromarray(image).convert('RGB')]
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(device, dtype=vl_gpt.dtype)
        
        # Generate response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=vl_gpt.prepare_inputs_embeds(**prepare_inputs),
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            use_cache=True
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        logger.error(f"Understanding error: {str(e)}")
        return f"Error processing request: {str(e)}"

@torch.inference_mode()
def generate_image(prompt, seed=12345, guidance=5.0, temperature=1.0, parallel_size=4):
    """Handle image generation requests"""
    try:
        # Text processing
        messages = [{'role': '<|User|>', 'content': prompt}, {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        ) + vl_chat_processor.image_start_tag
        
        # Generate image tokens
        input_ids = torch.LongTensor(tokenizer.encode(text)).to(device)
        generated_tokens, patches = generate(
            input_ids=input_ids,
            width=384,
            height=384,
            cfg_weight=guidance,
            parallel_size=parallel_size,
            temperature=temperature
        )
        
        # Process output images
        images = unpack(patches, 384, 384, parallel_size)
        return [Image.fromarray(img).resize((768, 768), Image.Resampling.LANCZOS) for img in images]
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return []

def generate(input_ids, width, height, **kwargs):
    """Core image generation function"""
    try:
        parallel_size = kwargs.get('parallel_size', 4)
        image_token_num_per_image = 576
        
        # Initialize tokens
        tokens = torch.stack([input_ids] * (parallel_size * 2), dim=0)
        generated = torch.zeros((parallel_size, image_token_num_per_image), 
                              dtype=torch.int, device=device)
        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        
        pkv = None
        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds,
                past_key_values=pkv,
                use_cache=True
            )
            pkv = outputs.past_key_values
            logits = vl_gpt.gen_head(outputs.last_hidden_state[:, -1, :])
            
            # Classifier-free guidance
            logit_cond, logit_uncond = logits[0::2], logits[1::2]
            logits = logit_uncond + kwargs['cfg_weight'] * (logit_cond - logit_uncond)
            
            # Sampling
            probs = torch.softmax(logits / kwargs['temperature'], dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated[:, i] = next_token.squeeze()
            
            # Prepare next input
            inputs_embeds = vl_gpt.prepare_gen_img_embeds(
                next_token.repeat(1, 2).view(-1)
            ).unsqueeze(1)

        # Decode patches
        return generated, vl_gpt.gen_vision_model.decode_code(
            generated.to(torch.int),
            shape=[parallel_size, 8, width//16, height//16]
        )
    
    except Exception as e:
        logger.error(f"Generate core error: {str(e)}")
        raise

def unpack(dec, width, height, parallel_size):
    """Convert model output to images"""
    try:
        dec = dec.float().cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) * 127.5, 0, 255).astype(np.uint8)
        return [dec[i] for i in range(parallel_size)]
    except Exception as e:
        logger.error(f"Unpack error: {str(e)}")
        return [np.zeros((height, width, 3), dtype=np.uint8)] * parallel_size

# Gradio Interface
with gr.Blocks(title="Janus Pro 7B", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖼️ Janus Pro 7B - Multimodal AI Assistant")
    
    with gr.Tab("Image Understanding"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy")
                # examples_und = gr.Examples(
                #     examples=[
                #         ["explain this meme", "images/doge.png"],
                #         ["Convert the formula into latex code", "images/equation.png"]
                #     ],
                #     inputs=[gr.Textbox(), image_input],  # Use component references
                #     label="Example Queries"
                # )
            with gr.Column():
                question_input = gr.Textbox(label="Question", placeholder="Ask about the image...")
                with gr.Accordion("Advanced Settings", open=False):
                    und_seed = gr.Number(42, label="Seed", precision=0)
                    top_p = gr.Slider(0, 1, 0.95, label="Top-p Sampling")
                    temperature = gr.Slider(0, 1, 0.1, label="Temperature")
                    max_tokens = gr.Slider(128, 2048, 1024, step=128, label="Max Tokens")
                understanding_button = gr.Button("Analyze", variant="primary")
                understanding_output = gr.Textbox(label="Response", interactive=False)

    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", placeholder="Describe your image...", lines=3)
                examples_t2i = gr.Examples(
                    examples=[
                        "Master shifu raccoon wearing streetwear",
                        "Astronaut in a jungle, detailed 8k rendering"
                    ],
                    inputs=prompt_input,
                    label="Example Prompts"
                )
                with gr.Accordion("Advanced Settings", open=False):
                    cfg_weight = gr.Slider(1, 10, 5.0, label="CFG Weight")
                    t2i_temp = gr.Slider(0, 2, 1.0, label="Temperature")
                    seed_input = gr.Number(12345, label="Seed", precision=0)
                    parallel_size = gr.Slider(1, 8, 4, step=1, label="Batch Size")
                generation_button = gr.Button("Generate", variant="primary")
            with gr.Column():
                image_output = gr.Gallery(label="Generated Images", columns=2, height=600)

    # Event handlers
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed, top_p, temperature, max_tokens],
        outputs=understanding_output
    )
    
    generation_button.click(
        generate_image,
        inputs=[prompt_input, seed_input, cfg_weight, t2i_temp, parallel_size],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=2).launch(
        server_name="127.0.0.1",
        server_port=7920,
        share=False
    )
