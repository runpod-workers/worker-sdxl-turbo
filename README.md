<div align="center">

<h1>SDXL Turbo Worker Template</h1>

A specialized worker template for building custom RunPod Endpoint API workers utilizing the SDXL Turbo model.

</div>

## Example Input

```json
{
    "input": {
        "prompt": "An image of a cat with a hat on.",
    }
}
```

## Example Output

The output from the SDXL Turbo Worker is a base64 encoded string of the generated image. Below is an example of what this output might look like:

data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...


To view the image, you can decode this base64 string into an image file using a suitable tool or programming library. For instance, in Python, you can use the following snippet:

```python
import base64
from PIL import Image
import io

# Replace 'base64_string' with your actual base64 string
base64_string = "iVBORw0KGgoAAAANSUhEUgAA..."
image_data = base64.b64decode(base64_string)
image = Image.open(io.BytesIO(image_data))
image.show()
```
