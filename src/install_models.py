from diffusers.models import AutoencoderKL
from diffusers import AutoPipelineForText2Image
import torch

model_name='SG161222/RealVisXL_V4.0'
#model_name='SG161222/RealVisXL_V5.0'

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
pipe = AutoPipelineForText2Image.from_pretrained(
    model_name,
    vae=vae,
    use_safetensors=True,
    add_watermarker=False,
    custom_pipeline="lpw_stable_diffusion_xl"
)
