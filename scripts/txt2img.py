import os
import sys

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from huggingface_hub import login

assert torch.cuda.is_available()

login(token=os.environ['HF_TOKEN'])

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16"
)
pipe = pipe.to("cuda")

prompt = sys.argv[1]
image = pipe(
    prompt,
    width=640,
    height=640,
    guidance_scale=7.5, num_inference_steps=120
).images[0]

image.save("/outputs/img.png")
