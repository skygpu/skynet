#!/usr/bin/python

import time
import random

from typing import Optional
from pathlib import Path

import torch
import numpy as np

from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from realesrgan import RealESRGANer
from huggingface_hub import login

from .constants import ALGOS


def time_ms():
    return int(time.time() * 1000)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)


def pipeline_for(algo: str, mem_fraction: float = 1.0, image=False):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(mem_fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    params = {
        'torch_dtype': torch.float16,
        'safety_checker': None
    }

    if algo == 'stable':
        params['revision'] = 'fp16'

    if image:
        pipe_class = StableDiffusionImg2ImgPipeline
    else:
        pipe_class = StableDiffusionPipeline

    pipe = pipe_class.from_pretrained(
        ALGOS[algo], **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    if not image:
        pipe.enable_vae_slicing()

    return pipe.to('cuda')


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


def img2img(
    hf_token: str,
    model: str = 'midj',
    prompt: str = 'a red old tractor in a sunny wheat field',
    img_path: str = 'input.png',
    output: str = 'output.png',
    strength: float = 1.0,
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
    pipe = pipeline_for(model, image=True)

    input_img = Image.open(img_path).convert('RGB')

    seed = seed if seed else random.randint(0, 2 ** 64)
    prompt = prompt
    image = pipe(
        prompt,
        image=input_img,
        strength=strength,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]

    image.save(output)


def upscale(
    img_path: str = 'input.png',
    output: str = 'output.png',
    model_path: str = 'weights/RealESRGAN_x4plus.pth'
):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    input_img = Image.open(img_path).convert('RGB')

    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        ),
        half=True)

    up_img, _ = upscaler.enhance(
        convert_from_image_to_cv2(input_img), outscale=4)

    image = convert_from_cv2_to_image(up_img)


    image.save(output)


def download_all_models(hf_token: str):
    assert torch.cuda.is_available()

    login(token=hf_token)
    for model in ALGOS:
        print(f'DOWNLOADING {model.upper()}')
        pipeline_for(model)

