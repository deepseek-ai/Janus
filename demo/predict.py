from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from janus.models import VLChatProcessor
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_CACHE = "checkpoints"
# MODEL_URL = "https://weights.replicate.delivery/default/deepseek-ai/Janus-Pro-1B/model.tar"
MODEL_URL = "https://weights.replicate.delivery/default/deepseek-ai/Janus-Pro-7B/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        config = AutoConfig.from_pretrained(MODEL_CACHE)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            language_config=language_config,
            trust_remote_code=True
        )
        
        if torch.cuda.is_available():
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda()
        else:
            self.vl_gpt = self.vl_gpt.to(torch.float16)

        self.vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_CACHE)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image for multimodal understanding"),
        question: str = Input(description="Question about the image"),
        seed: int = Input(description="Random seed for reproducibility", default=42),
        top_p: float = Input(description="Top-p sampling value", default=0.95, ge=0, le=1),
        temperature: float = Input(description="Temperature for text generation", default=0.1, ge=0, le=1),
    ) -> str:
        """Run a single prediction on the model"""
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Load and process image
        pil_image = Image.open(image)
        image_array = np.array(pil_image)
        
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [image_array],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = [pil_image]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self.cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
        
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False if temperature == 0 else True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer 