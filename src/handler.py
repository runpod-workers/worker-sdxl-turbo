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

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
#enhanceaiteam/Flux-uncensored
try:
    HF_TOKEN = os.getenv("HF_TOKEN")
    print("HFToken ################# : ", HF_TOKEN)
    # pipe = AutoPipelineForText2Image.from_pretrained("enhanceaiteam/Flux-uncensored", torch_dtype=torch.float16, variant="fp16")
    # pipe.to("cuda")
    pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", token=HF_TOKEN)
    pipe.load_lora_weights("enhanceaiteam/Flux-uncensored")
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
