#!/usr/bin/python

import random

from typing import Optional
from pathlib import Path

import torch

from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    EulerAncestralDiscreteScheduler
)
from huggingface_hub import login

from .dgpu import pipeline_for


def txt2img(
    hf_token: str,
    model: str = 'midj',
    prompt: str = 'a red old tractor in a sunny wheat field',
    output: str = 'output.png',
    width: int = 512, height: int = 512,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    login(token=hf_token)
    pipe = pipeline_for(model)

    seed = seed if seed else random.randint(0, 2 ** 64)
    prompt = prompt
    image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]

    image.save(output)


def upscale(
    hf_token: str,
    prompt: str = 'a red old tractor in a sunny wheat field',
    img_path: str = 'input.png',
    output: str = 'output.png',
    steps: int = 28
):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    login(token=hf_token)
    params = {
        'torch_dtype': torch.float16,
        'safety_checker': None
    }

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        'stabilityai/stable-diffusion-x4-upscaler', **params)

    prompt = prompt
    image = pipe(
        prompt,
        image=Image.open(img_path)
    ).images[0]

    image.save(output)
