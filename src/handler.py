""" Example handler file. """

import runpod
from diffusers.models import AutoencoderKL
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
import base64
import io
import sys
import time
import subprocess
import os
import tempfile
import numpy as np
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from typing import Callable, Dict, Optional, Tuple

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.





def seed_everything(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def get_scheduler(scheduler_config: Dict, name: str) -> Optional[Callable]:
    scheduler_factory_map = {
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ SDE Karras": lambda: DPMSolverSinglestepScheduler.from_config(scheduler_config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(scheduler_config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
        "Euler": lambda: EulerDiscreteScheduler.from_config(scheduler_config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(scheduler_config),
        "DDIM": lambda: DDIMScheduler.from_config(scheduler_config),
    }
    return scheduler_factory_map.get(name, lambda: None)()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    num_inference_steps = job_input['num_inference_steps']
    guidance_scale = job_input['guidance_scale']
    negative_prompt = job_input['negative_prompt']
    width = job_input['width']
    height = job_input['height']
    seed = job_input['seed']
    sampler = job_input.get('sampler','DPM++ 2M Karras')
    model_name = job_input.get('model_name', "SG161222/RealVisXL_V4.0")  # Default model name if not provided

    generator = seed_everything(seed)

    time_start = time.time()

    # Load model dynamically based on model_name
    try:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16"
        )
        pipe.scheduler = get_scheduler(pipe.scheduler.config, sampler)
        pipe.to("cuda")
    except RuntimeError as e:
        return {"error": f"Failed to load model: {e}"}





    image = pipe(prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
	num_inference_steps=num_inference_steps,
	guidance_scale=guidance_scale,
        generator=generator
	).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')



def scriptHandler(event):
    try:
        script = event.get("input", {}).get("script", "")
        if not script:
            return {"error": "No script provided"}

        # Redirect stdout to capture print output
        stdout_backup = sys.stdout
        sys.stdout = io.StringIO()

        # Create a shared dictionary to capture JSON output
        exec_globals = {"event": event, "output": {}}

        try:
            exec(script, exec_globals)  # Execute script with access to `event` and `output`
            output_json = exec_globals["output"]  # Extract JSON output
            output_text = sys.stdout.getvalue()  # Extract printed output
        finally:
            sys.stdout = stdout_backup  # Restore original stdout

        return {
            "stdout": output_text.strip(),
            "output": output_json,  # JSON output from script
            "stderr": "",
            "return_code": 0
        }

    except Exception as e:
        return {"error": str(e)}


def handler(event):
    try:
        script = event.get("script", "")
        if not script:
            return {"error": "No script provided"}

        # Redirect stdout to capture script output
        stdout_backup = sys.stdout
        sys.stdout = io.StringIO()

        # Create a safe execution environment with access to `event`
        exec_globals = {"event": event}  # Pass `event` as a global variable

        try:
            exec(script, exec_globals)  # Run the script with access to `event`
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = stdout_backup  # Restore original stdout

        return {"stdout": output, "stderr": "", "return_code": 0}

    except Exception as e:
        return {"error": str(e)}



#runpod.serverless.start({"handler": handler})
runpod.serverless.start({"handler": scriptHandler})
