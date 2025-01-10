#!/usr/bin/python

# Skynet Memory Manager

import gc
import logging

from hashlib import sha256
import zipfile
from PIL import Image
from diffusers import DiffusionPipeline

import trio
import torch

from skynet.constants import DEFAULT_INITAL_MODEL, MODELS
from skynet.dgpu.errors import DGPUComputeError, DGPUInferenceCancelled

from skynet.utils import crop_image, convert_from_cv2_to_image, convert_from_image_to_cv2, convert_from_img_to_bytes, init_upscaler, pipeline_for


def prepare_params_for_diffuse(
    params: dict,
    mode: str,
    inputs: list[bytes]
):
    _params = {}
    match mode:
        case 'inpaint':
            image = crop_image(
                inputs[0], params['width'], params['height'])

            mask = crop_image(
                inputs[1], params['width'], params['height'])

            _params['image'] = image
            _params['mask_image'] = mask
            _params['strength'] = float(params['strength'])

        case 'img2img':
            image = crop_image(
                inputs[0], params['width'], params['height'])

            _params['image'] = image
            _params['strength'] = float(params['strength'])

        case 'txt2img' | 'diffuse':
            ...

        case _:
            raise DGPUComputeError(f'Unknown mode {mode}')

    # _params['width'] = int(params['width'])
    # _params['height'] = int(params['height'])

    return (
        params['prompt'],
        float(params['guidance']),
        int(params['step']),
        torch.manual_seed(int(params['seed'])),
        params['upscaler'] if 'upscaler' in params else None,
        _params
    )


class SkynetMM:

    def __init__(self, config: dict):
        self.upscaler = init_upscaler()

        self.cache_dir = None
        if 'hf_home' in config:
            self.cache_dir = config['hf_home']

        self._model_name = ''
        self._model_mode = ''

        # self.load_model(DEFAULT_INITAL_MODEL, 'txt2img')

    def log_debug_info(self):
        logging.info('memory summary:')
        logging.info('\n' + torch.cuda.memory_summary())

    def is_model_loaded(self, name: str, mode: str):
        if (name == self._model_name and
            mode == self._model_mode):
            return True

        return False

    def load_model(
        self,
        name: str,
        mode: str
    ):
        logging.info(f'loading model {name}...')
        self._model_mode = mode
        self._model_name = name

        if getattr(self, '_model', None):
            del self._model

        gc.collect()
        torch.cuda.empty_cache()

        self._model = pipeline_for(
            name, mode, cache_dir=self.cache_dir)

    def get_model(self, name: str, mode: str) -> DiffusionPipeline:
        if name not in MODELS:
            raise DGPUComputeError(f'Unknown model {model_name}')

        if not self.is_model_loaded(name, mode):
            self.load_model(name, mode)

    def compute_one(
        self,
        request_id: int,
        method: str,
        params: dict,
        inputs: list[bytes] = []
    ):
        def maybe_cancel_work(step, *args, **kwargs):
            if self._should_cancel:
                should_raise = trio.from_thread.run(self._should_cancel, request_id)
                if should_raise:
                    logging.warn(f'cancelling work at step {step}')
                    raise DGPUInferenceCancelled()

        maybe_cancel_work(0)

        output_type = 'png'
        if 'output_type' in params:
            output_type = params['output_type']

        output = None
        output_hash = None
        try:
            match method:
                case 'diffuse' | 'txt2img' | 'img2img' | 'inpaint':
                    arguments = prepare_params_for_diffuse(
                        params, method, inputs)
                    prompt, guidance, step, seed, upscaler, extra_params = arguments
                    self.get_model(
                        params['model'],
                        method
                    )

                    output = self._model(
                        prompt,
                        guidance_scale=guidance,
                        num_inference_steps=step,
                        generator=seed,
                        callback=maybe_cancel_work,
                        callback_steps=1,
                        **extra_params
                    ).images[0]

                    output_binary = b''
                    match output_type:
                        case 'png':
                            if upscaler == 'x4':
                                input_img = output.convert('RGB')
                                up_img, _ = self.upscaler.enhance(
                                    convert_from_image_to_cv2(input_img), outscale=4)

                                output = convert_from_cv2_to_image(up_img)

                            output_binary = convert_from_img_to_bytes(output)

                        case _:
                            raise DGPUComputeError(f'Unsupported output type: {output_type}')

                    output_hash = sha256(output_binary).hexdigest()

                case _:
                    raise DGPUComputeError('Unsupported compute method')

        except BaseException as e:
            logging.error(e)
            raise DGPUComputeError(str(e))

        finally:
            torch.cuda.empty_cache()

        return output_hash, output
