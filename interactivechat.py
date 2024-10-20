import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import time
import re

# Specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def create_prompt(user_input: str) -> str:
    conversation = [
        {
            "role": "User",
            "content": user_input,
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    short_prompt: str,
    parallel_size: int = 16,
    temperature: float = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    outputs = None  # Initialize outputs for use in the loop

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)

    # Create a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Sanitize the short_prompt to ensure it's safe for filenames
    short_prompt = re.sub(r'\W+', '_', short_prompt)[:50]

    # Save images with timestamp and part of the user prompt in the filename
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', f"img_{timestamp}_{short_prompt}_{i}.jpg")
        PIL.Image.fromarray(visual_img[i]).save(save_path)


def interactive_image_generator():
    print("Welcome to the interactive image generator!")

    # Ask for the number of images at the start of the session
    while True:
        num_images_input = input("How many images would you like to generate per prompt? (Enter a positive integer): ")
        if num_images_input.isdigit() and int(num_images_input) > 0:
            parallel_size = int(num_images_input)
            break
        else:
            print("Invalid input. Please enter a positive integer.")

    while True:
        user_input = input("Please describe the image you'd like to generate (or type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the image generator. Goodbye!")
            break

        prompt = create_prompt(user_input)

        # Create a sanitized version of user_input for the filename
        short_prompt = re.sub(r'\W+', '_', user_input)[:50]

        print(f"Generating {parallel_size} image(s) for: '{user_input}'")
        generate(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt,
            short_prompt=short_prompt,
            parallel_size=parallel_size  # Pass the user-specified number of images
        )

        print("Image generation complete! Check the 'generated_samples' folder for the output.\n")


if __name__ == "__main__":
    interactive_image_generator()
