""" Example handler file. """
## import library
import os
import runpod
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline

import torch
import base64
import io
import time
from cryptography.fernet import Fernet
# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    value1 = "gAAAAABoXBJ0PUmtUdNYDzLc9aK9i3cPiwTrcPGhFu1DTcKtV-bfcb0yKYtHoPjVl1MivWv9J-sMO2wv8ayFlqx0bDBzl0F0XSacfiJomLdcJHLBe07u8xEihRV8sQca_4kWgNWQFcAh"
    value2= "HCvCU3FTgiDFIbyYkMR5qILRvvdwCq_bjfVEZwj1m8Q="
    value3 = Fernet(value2.encode()).decrypt(value1).decode()

    print("print checkpoint ################# : ", value3)
    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", token=value3, torch_dtype=torch.bfloat16).to('cuda')
    pipe.load_lora_weights('enhanceaiteam/Flux-uncensored', weight_name='lora.safetensors')
    
    # pipe = AutoPipelineForText2Image.from_pretrained("enhanceaiteam/Flux-uncensored", torch_dtype=torch.float16, variant="fp16")
    
    # pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token=HF_TOKEN)
    # pipe.load_lora_weights("enhanceaiteam/Flux-uncensored")
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    time_start = time.time()
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=4).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
