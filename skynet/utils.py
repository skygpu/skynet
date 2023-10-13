#!/usr/bin/python

import io
import os
import sys
import time
import random
import logging

from typing import Optional
from pathlib import Path
import asks

import torch
import numpy as np

from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler
)
from realesrgan import RealESRGANer
from huggingface_hub import login
import trio

from .constants import MODELS


def time_ms():
    return int(time.time() * 1000)


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)


def convert_from_bytes_to_img(raw: bytes) -> Image:
    return Image.open(io.BytesIO(raw))


def convert_from_img_to_bytes(image: Image, fmt='PNG') -> bytes:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=fmt)
    return byte_arr.getvalue()


def crop_image(image: Image, max_w: int, max_h: int) -> Image:
    w, h = image.size
    if w > max_w or h > max_h:
        image.thumbnail((max_w, max_h))

    return image.convert('RGB')


def pipeline_for(
    model: str,
    mem_fraction: float = 1.0,
    image: bool = False,
    cache_dir: str | None = None
) -> DiffusionPipeline:

    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # full determinism
    # https://huggingface.co/docs/diffusers/using-diffusers/reproducibility#deterministic-algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    model_info = MODELS[model]

    req_mem = model_info['mem']
    mem_gb = torch.cuda.mem_get_info()[1] / (10**9)
    mem_gb *= mem_fraction
    over_mem = mem_gb < req_mem
    if over_mem:
        logging.warn(f'model requires {req_mem} but card has {mem_gb}, model will run slower..')

    shortname = model_info['short']

    params = {
        'safety_checker': None,
        'torch_dtype': torch.float16,
        'cache_dir': cache_dir,
        'variant': 'fp16'
    }

    match shortname:
        case 'stable':
            params['revision'] = 'fp16'

    torch.cuda.set_per_process_memory_fraction(mem_fraction)

    pipe = DiffusionPipeline.from_pretrained(
        model, **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    if over_mem:
        if not image:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()

        pipe.enable_model_cpu_offload()

    else:
        if sys.version_info[1] < 11:
            # torch.compile only supported on python < 3.11
            pipe.unet = torch.compile(
                pipe.unet, mode='reduce-overhead', fullgraph=True)

        pipe = pipe.to('cuda')

    return pipe


def txt2img(
    hf_token: str,
    model: str = 'prompthero/openjourney',
    prompt: str = 'a red old tractor in a sunny wheat field',
    output: str = 'output.png',
    width: int = 512, height: int = 512,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
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
    model: str = 'prompthero/openjourney',
    prompt: str = 'a red old tractor in a sunny wheat field',
    img_path: str = 'input.png',
    output: str = 'output.png',
    strength: float = 1.0,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
    login(token=hf_token)
    pipe = pipeline_for(model, image=True)

    with open(img_path, 'rb') as img_file:
        input_img = convert_from_bytes_and_crop(img_file.read(), 512, 512)

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


def init_upscaler(model_path: str = 'weights/RealESRGAN_x4plus.pth'):
    return RealESRGANer(
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
        half=True
    )

def upscale(
    img_path: str = 'input.png',
    output: str = 'output.png',
    model_path: str = 'weights/RealESRGAN_x4plus.pth'
):
    input_img = Image.open(img_path).convert('RGB')

    upscaler = init_upscaler(model_path=model_path)

    up_img, _ = upscaler.enhance(
        convert_from_image_to_cv2(input_img), outscale=4)

    image = convert_from_cv2_to_image(up_img)
    image.save(output)


async def download_upscaler():
    print('downloading upscaler...')
    weights_path = Path('weights')
    weights_path.mkdir(exist_ok=True)
    upscaler_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    save_path = weights_path / 'RealESRGAN_x4plus.pth'
    response = await asks.get(upscaler_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print('done')

def download_all_models(hf_token: str, hf_home: str):
    assert torch.cuda.is_available()

    trio.run(download_upscaler)

    login(token=hf_token)
    for model in MODELS:
        print(f'DOWNLOADING {model.upper()}')
        pipeline_for(model, cache_dir=hf_home)
