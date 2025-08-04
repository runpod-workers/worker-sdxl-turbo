INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "negative_prompt": {"type": str, "required": False, "default": None},
    "height": {
        "type": int,
        "required": False,
        "default": 512,
        "constraints": lambda height: 64 <= height <= 1024 and height % 8 == 0,
    },
    "width": {
        "type": int,
        "required": False,
        "default": 512,
        "constraints": lambda width: 64 <= width <= 1024 and width % 8 == 0,
    },
    "seed": {"type": int, "required": False, "default": None},
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda steps: 1 <= steps <= 8,
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 0.0,
        "constraints": lambda scale: 0.0 <= scale <= 2.0,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda img_count: 5 > img_count > 0,
    },
}
