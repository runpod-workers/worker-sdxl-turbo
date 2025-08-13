![SDXL Turbo Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl-turbo_banner-placeholder.jpeg)

---

Run [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) as a serverless endpoint for ultra-fast image generation.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl-turbo)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl-turbo)

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type    | Default | Required | Description                                                       |
| :-------------------- | :------ | :------ | :------- | :---------------------------------------------------------------- |
| `prompt`              | `str`   | `None`  | **Yes**  | The main text prompt describing the desired image                 |
| `negative_prompt`     | `str`   | `None`  | No       | Text prompt specifying concepts to exclude from the image         |
| `height`              | `int`   | `512`   | No       | The height of the generated image in pixels (must be 512)         |
| `width`               | `int`   | `512`   | No       | The width of the generated image in pixels (must be 512)          |
| `num_inference_steps` | `int`   | `1`     | No       | Number of denoising steps (1-8 for optimal Turbo performance)     |
| `guidance_scale`      | `float` | `0.0`   | No       | Guidance scale (typically 0.0 for Turbo, up to 2.0)               |
| `seed`                | `int`   | `None`  | No       | Random seed for reproducibility. If `None`, a random seed is used |
| `num_images`          | `int`   | `1`     | No       | Number of images to generate per prompt (Constraint: must be 1-4) |

### Example Request

```json
{
  "input": {
    "prompt": "a majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly",
    "height": 512,
    "width": 512,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 1337,
    "num_images": 1
  }
}
```

### Example Response

which is producing an output like this:

```json
{
  "delayTime": 2134,
  "executionTime": 1247,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "images": [
      {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
        "seed": 1337
      }
    ],
    "generation_time": 0.85,
    "parameters": {
      "prompt": "a majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
      "negative_prompt": "blurry, low quality, deformed, ugly",
      "width": 512,
      "height": 512,
      "num_inference_steps": 4,
      "guidance_scale": 0.0,
      "seed": 1337
    }
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```
