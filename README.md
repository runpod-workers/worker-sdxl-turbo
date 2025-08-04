![SDXL Turbo Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl-turbo_banner-placeholder.jpeg)

---

Run [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) as a serverless endpoint for ultra-fast image generation.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl-turbo)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl-turbo)

---

## Features

- ⚡ **Ultra-fast generation**: 1-4 inference steps vs 25+ for regular SDXL
- 🎯 **High quality output**: Maintains excellent image quality despite speed
- 🔧 **Flexible parameters**: Customizable dimensions, steps, and seeds
- 📦 **Production ready**: Full input validation, error handling, and monitoring
- 🐳 **Containerized**: Ready for deployment on RunPod serverless infrastructure

## Model Information

This worker uses **Stable Diffusion XL Turbo** from Stability AI, which is an adversarially trained distilled version of SDXL that can generate high-quality images in just 1-4 inference steps.

## API Reference

### Input Schema

| Parameter             | Type    | Required | Default | Description                                       |
| --------------------- | ------- | -------- | ------- | ------------------------------------------------- |
| `prompt`              | string  | ✅       | -       | Text description of the image to generate         |
| `negative_prompt`     | string  | ❌       | null    | What to avoid in the image                        |
| `height`              | integer | ❌       | 512     | Image height (64-1024, must be divisible by 8)    |
| `width`               | integer | ❌       | 512     | Image width (64-1024, must be divisible by 8)     |
| `num_inference_steps` | integer | ❌       | 1       | Number of denoising steps (1-8)                   |
| `guidance_scale`      | float   | ❌       | 0.0     | Guidance scale (0.0-2.0, typically 0.0 for Turbo) |
| `seed`                | integer | ❌       | random  | Seed for reproducible generation                  |
| `num_images`          | integer | ❌       | 1       | Number of images to generate (1-4)                |

### Example Request

```json
{
  "input": {
    "prompt": "a majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 1337,
    "num_images": 1
  }
}
```

### Example Response

```json
{
  "images": [
    {
      "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
      "seed": 1337
    }
  ],
  "generation_time": 0.85,
  "parameters": {
    "prompt": "a majestic steampunk dragon...",
    "negative_prompt": "blurry, low quality...",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 1337
  }
}
```

## Performance

- **Generation time**: ~0.5-2 seconds per image (depending on size and steps)
- **Memory usage**: ~6GB VRAM for 512x512, ~12GB for 1024x1024
- **Recommended GPU**: RTX 4090, A100, or similar

## Development

### Local Testing

```bash
# Download model weights
python download_weights.py

# Test handler locally
python handler.py --test_input='{"prompt": "a cat wearing a hat"}'
```

### Building the Docker Image

```bash
# Build for AMD64
docker build --platform linux/amd64 -t sdxl-turbo-worker .

# Run locally
docker run --gpus all -p 8000:8000 sdxl-turbo-worker
```

## Deployment

This worker is designed to be deployed on RunPod's serverless infrastructure. The `.runpod` directory contains all necessary configuration files for automatic deployment.

### Requirements

- GPU with at least 8GB VRAM
- CUDA 12.1+ support
- Container disk space: 15GB minimum

## Tips for Best Results

1. **Use guidance_scale=0.0**: SDXL Turbo works best without guidance
2. **Keep steps low**: 1-4 steps are optimal; more steps may degrade quality
3. **Optimal sizes**: 512x512 for speed, 1024x1024 for quality
4. **Clear prompts**: Be specific and descriptive for better results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
