"""
SDXL Turbo Worker for RunPod Serverless
Ultra-fast image generation with Stable Diffusion XL Turbo
"""

import os
import base64
import io
import time
from typing import Optional, Dict, Any

import torch
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from diffusers import AutoPipelineForText2Image
from PIL import Image

from schemas import INPUT_SCHEMA


class ModelHandler:
    def __init__(self):
        """Initialize the SDXL Turbo pipeline."""
        self.pipe = None
        self.load_model()

    def load_model(self):
        """Load the SDXL Turbo model."""
        print("🚀 Loading SDXL Turbo model...")

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=False,
            )

            if torch.cuda.is_available():
                self.pipe.to("cuda")
                print("✅ Model loaded successfully on GPU!")
            else:
                print("⚠️  GPU not available, running on CPU")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load SDXL Turbo model: {str(e)}")

    def generate_image(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using SDXL Turbo."""

        # Extract parameters
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")
        height = job_input.get("height", 512)
        width = job_input.get("width", 512)
        num_inference_steps = job_input.get("num_inference_steps", 1)
        guidance_scale = job_input.get("guidance_scale", 0.0)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")

        print(f"🎨 Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
        print(
            f"📐 Size: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}"
        )

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        try:
            start_time = time.time()

            # Generate images
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
            )

            generation_time = time.time() - start_time
            print(f"⚡ Generated in {generation_time:.2f} seconds")

            # Process images
            images_data = []
            for i, image in enumerate(result.images):
                # Convert to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                images_data.append(
                    {"image": image_b64, "seed": seed + i if seed is not None else None}
                )

            return {
                "images": images_data,
                "generation_time": generation_time,
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                },
            }

        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")


# Initialize model handler
model_handler = ModelHandler()


def handler(job):
    """
    Handler function for RunPod serverless.
    """
    try:
        # Validate input
        job_input = job["input"]

        # Validate against schema
        validated_input = validate(job_input, INPUT_SCHEMA)
        if "errors" in validated_input:
            return {"error": f"Input validation failed: {validated_input['errors']}"}

        validated_data = validated_input["validated_input"]

        # Generate image
        result = model_handler.generate_image(validated_data)

        return result

    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("🎯 Starting SDXL Turbo Worker...")
    runpod.serverless.start({"handler": handler})
