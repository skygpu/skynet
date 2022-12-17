#!/usr/bin/python

import random

from typing import Optional
from pathlib import Path

import torch

from diffusers import StableDiffusionPipeline
from huggingface_hub import login


def txt2img(
    hf_token: str,
    model_name: str,
    prompt: str,
    output: str,
    width: int, height: int,
    guidance: float,
    steps: int,
    seed: Optional[int]
):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.333)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    login(token=hf_token)

    params = {
        'torch_dtype': torch.float16,
        'safety_checker': None
    }
    if model_name == 'runwayml/stable-diffusion-v1-5':
        params['revision'] = 'fp16'

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    pipe = pipe.to("cuda")

    seed = ireq.seed if ireq.seed else random.randint(0, 2 ** 64)
    prompt = prompt
    image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]

    image.save(output)
