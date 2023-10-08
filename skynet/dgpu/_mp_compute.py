#!/usr/bin/python

import gc
import json
import logging

from hashlib import sha256
from diffusers import DiffusionPipeline

import trio
import torch

from skynet.constants import DEFAULT_INITAL_MODELS, MODELS
from skynet.dgpu.errors import DGPUComputeError

from skynet.utils import convert_from_bytes_and_crop, convert_from_cv2_to_image, convert_from_image_to_cv2, convert_from_img_to_bytes, init_upscaler, pipeline_for


def prepare_params_for_diffuse(
    params: dict,
    binary: bytes | None = None
):
    image = None
    if binary:
        image = convert_from_bytes_and_crop(binary, 512, 512)

    _params = {}
    if image:
        _params['image'] = image
        _params['strength'] = float(params['strength'])

    else:
        _params['width'] = int(params['width'])
        _params['height'] = int(params['height'])

    return (
        params['prompt'],
        float(params['guidance']),
        int(params['step']),
        torch.manual_seed(int(params['seed'])),
        params['upscaler'] if 'upscaler' in params else None,
        _params
    )


_models = {}

def is_model_loaded(model_name: str, image: bool):
    for model_key, model_data in _models.items():
        if (model_key == model_name and
            model_data['image'] == image):
            return True

    return False

def load_model(
    model_name: str,
    image: bool,
    force=False
):
    logging.info(f'loading model {model_name}...')
    if force or len(_models.keys()) == 0:
        pipe = pipeline_for(
            model_name, image=image)

        _models[model_name] = {
            'pipe': pipe,
            'generated': 0,
            'image': image
        }

    else:
        least_used = list(_models.keys())[0]

        for model in _models:
            if _models[
                least_used]['generated'] > _models[model]['generated']:
                least_used = model

        del _models[least_used]

        logging.info(f'swapping model {least_used} for {model_name}...')

        gc.collect()
        torch.cuda.empty_cache()

        pipe = pipeline_for(
            model_name, image=image)

        _models[model_name] = {
            'pipe': pipe,
            'generated': 0,
            'image': image
        }

    logging.info(f'loaded model {model_name}')
    return pipe

def get_model(model_name: str, image: bool) -> DiffusionPipeline:
    if model_name not in MODELS:
        raise DGPUComputeError(f'Unknown model {model_name}')

    if not is_model_loaded(model_name, image):
        pipe = load_model(model_name, image=image)

    else:
        pipe = _models[model_name]['pipe']

    return pipe

def _static_compute_one(kwargs: dict):
    request_id: int = kwargs['request_id']
    method: str = kwargs['method']
    params: dict = kwargs['params']
    binary: bytes | None = kwargs['binary']

    def _checkpoint(*args, **kwargs):
        trio.from_thread.run(trio.sleep, 0)    

    try:
        match method:
            case 'diffuse':
                image = None

                arguments = prepare_params_for_diffuse(params, binary)
                prompt, guidance, step, seed, upscaler, extra_params = arguments
                model = get_model(params['model'], 'image' in extra_params)

                image = model(
                    prompt,
                    guidance_scale=guidance,
                    num_inference_steps=step,
                    generator=seed,
                    callback=_checkpoint,
                    callback_steps=1,
                    **extra_params
                ).images[0]

                if upscaler == 'x4':
                    upscaler = init_upscaler()
                    input_img = image.convert('RGB')
                    up_img, _ = upscaler.enhance(
                        convert_from_image_to_cv2(input_img), outscale=4)

                    image = convert_from_cv2_to_image(up_img)

                img_raw = convert_from_img_to_bytes(image)
                img_sha = sha256(img_raw).hexdigest()

                return img_sha, img_raw

            case _:
                raise DGPUComputeError('Unsupported compute method')

    except BaseException as e:
        logging.error(e)
        raise DGPUComputeError(str(e))

    finally:
        torch.cuda.empty_cache()


async def _tractor_static_compute_one(**kwargs):
    return await trio.to_thread.run_sync(
        _static_compute_one, kwargs)
