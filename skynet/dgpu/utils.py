import io
import os
import sys
import time
import random
import logging
import importlib

from typing import Optional
from contextlib import contextmanager

import torch
import diffusers
import numpy as np

from PIL import Image
from diffusers import (
    DiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    EulerAncestralDiscreteScheduler,
)
from huggingface_hub import login

from skynet.constants import MODELS

# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks
# basicsr; see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer



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

def convert_from_bytes_and_crop(raw: bytes, max_w: int, max_h: int) -> Image:
    return crop_image(convert_from_bytes_to_img(raw), max_w, max_h)


class DummyPB:
    def update(self):
        ...

@torch.compiler.disable
@contextmanager
def dummy_progress_bar(*args, **kwargs):
    yield DummyPB()


def monkey_patch_pipeline_disable_progress_bar(pipe):
    pipe.progress_bar = dummy_progress_bar


def pipeline_for(
    model: str,
    mode: str,
    mem_fraction: float = 1.0,
    cache_dir: str | None = None
) -> DiffusionPipeline:
    diffusers.utils.logging.disable_progress_bar()

    logging.info(f'pipeline_for {model} {mode}')
    # assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # full determinism
    # https://huggingface.co/docs/diffusers/using-diffusers/reproducibility#deterministic-algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    model_info = MODELS[model]
    shortname = model_info.short

    # disable for compat with "diffuse" method
    # assert mode in model_info.tags

    # default to checking if custom pipeline exist and return that if not, attempt generic
    try:
        normalized_shortname = shortname.replace('-', '_')
        custom_pipeline = importlib.import_module(f'skynet.dgpu.pipes.{normalized_shortname}')
        assert custom_pipeline.__model['name'] == model
        pipe = custom_pipeline.pipeline_for(model, mode, mem_fraction=mem_fraction, cache_dir=cache_dir)
        monkey_patch_pipeline_disable_progress_bar(pipe)
        return pipe

    except ImportError:
        # TODO, uhh why not warn/error log this?
        ...


    req_mem = model_info.mem

    mem_gb = torch.cuda.mem_get_info()[1] / (10**9)
    mem_gb *= mem_fraction
    over_mem = mem_gb < req_mem
    if over_mem:
        logging.warn(f'model requires {req_mem} but card has {mem_gb}, model will run slower..')

    params = {
        'torch_dtype': torch.float16,
        'cache_dir': cache_dir,
        'variant': 'fp16',
    }

    match shortname:
        case 'stable':
            params['revision'] = 'fp16'
            params['safety_checker'] = None

    torch.cuda.set_per_process_memory_fraction(mem_fraction)

    pipe_class = DiffusionPipeline
    match mode:
        case 'inpaint':
            pipe_class = AutoPipelineForInpainting

        case 'img2img':
            pipe_class = AutoPipelineForImage2Image

        case 'txt2img':
            pipe_class = AutoPipelineForText2Image

    pipe = pipe_class.from_pretrained(
        model, **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    # pipe.enable_xformers_memory_efficient_attention()

    if over_mem:
        if mode == 'txt2img':
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        pipe.enable_model_cpu_offload()

    else:
        # if sys.version_info[1] < 11:
        #     # torch.compile only supported on python < 3.11
        #     pipe.unet = torch.compile(
        #         pipe.unet, mode='reduce-overhead', fullgraph=True)

        pipe = pipe.to('cuda')

    monkey_patch_pipeline_disable_progress_bar(pipe)

    return pipe


def txt2img(
    hf_token: str,
    model: str = list(MODELS.keys())[-1],
    prompt: str = 'a red old tractor in a sunny wheat field',
    output: str = 'output.png',
    width: int = 512, height: int = 512,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
    login(token=hf_token)
    pipe = pipeline_for(model, 'txt2img')

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
    model: str = list(MODELS.keys())[-2],
    prompt: str = 'a red old tractor in a sunny wheat field',
    img_path: str = 'input.png',
    output: str = 'output.png',
    strength: float = 1.0,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
    login(token=hf_token)
    pipe = pipeline_for(model, 'img2img')

    model_info = MODELS[model]

    with open(img_path, 'rb') as img_file:
        input_img = convert_from_bytes_and_crop(img_file.read(), model_info.size.w, model_info.size.h)

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


def inpaint(
    hf_token: str,
    model: str = list(MODELS.keys())[-3],
    prompt: str = 'a red old tractor in a sunny wheat field',
    img_path: str = 'input.png',
    mask_path: str = 'mask.png',
    output: str = 'output.png',
    strength: float = 1.0,
    guidance: float = 10,
    steps: int = 28,
    seed: Optional[int] = None
):
    login(token=hf_token)
    pipe = pipeline_for(model, 'inpaint')

    model_info = MODELS[model]

    with open(img_path, 'rb') as img_file:
        input_img = convert_from_bytes_and_crop(img_file.read(), model_info.size.w, model_info.size.h)

    with open(mask_path, 'rb') as mask_file:
        mask_img = convert_from_bytes_and_crop(mask_file.read(), model_info.size.w, model_info.size.h)

    var_params = {}
    if 'flux' not in model.lower():
        var_params['strength'] = strength

    seed = seed if seed else random.randint(0, 2 ** 64)
    prompt = prompt
    image = pipe(
        prompt,
        image=input_img,
        mask_image=mask_img,
        guidance_scale=guidance, num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        **var_params
    ).images[0]

    image.save(output)


def init_upscaler(model_path: str = 'hf_home/RealESRGAN_x4plus.pth'):
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
    model_path: str = 'hf_home/RealESRGAN_x4plus.pth'
):
    input_img = Image.open(img_path).convert('RGB')

    upscaler = init_upscaler(model_path=model_path)

    up_img, _ = upscaler.enhance(
        convert_from_image_to_cv2(input_img), outscale=4)

    image = convert_from_cv2_to_image(up_img)
    image.save(output)
