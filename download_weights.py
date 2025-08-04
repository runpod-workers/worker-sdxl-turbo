"""
Download and cache the SDXL Turbo model weights.
"""

import os
import torch
from diffusers import AutoPipelineForText2Image


def download_models():
    """Download and cache SDXL Turbo model."""
    print("Downloading SDXL Turbo model...")

    # Create cache directory if it doesn't exist
    cache_dir = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Download SDXL Turbo pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        print("✅ SDXL Turbo model downloaded successfully!")

        # Test loading to GPU if available
        if torch.cuda.is_available():
            pipe.to("cuda")
            print("✅ Model successfully loaded to GPU!")
        else:
            print("⚠️  GPU not available, model will run on CPU")

    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise


if __name__ == "__main__":
    download_models()
