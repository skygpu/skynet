#!/usr/bin/python
# ^TODO? again, why..
#
# Do we expect this mod
# to be invoked? if so why is there no
# `if __name__ == '__main__'` guard?
#
# if anything this should contain a license header ;)

'''
Skynet Memory Manager

'''

import gc
import logging

from hashlib import sha256
# import zipfile
# from PIL import Image
# from diffusers import DiffusionPipeline

import trio
import torch

# from skynet.constants import (
#     DEFAULT_INITAL_MODEL,
#     MODELS,
# )
from skynet.dgpu.errors import (
    DGPUComputeError,
    DGPUInferenceCancelled,
)

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

            if 'flux' in params['model'].lower():
                _params['max_sequence_length'] = 512
            else:
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


# TODO, yet again - drop the redundant prefix ;)
class SkynetMM:
    '''
    (AI algo) Model manager for loading models, computing outputs,
    checking load state, and unloading when no-longer-needed/finished.

    '''
    def __init__(self, config: dict):
        self.cache_dir = None
        if 'hf_home' in config:
            self.cache_dir = config['hf_home']

        self._model_name: str = ''
        self._model_mode: str = ''

        # self.load_model(DEFAULT_INITAL_MODEL, 'txt2img')

    def log_debug_info(self):
        logging.info('memory summary:')
        logging.info('\n' + torch.cuda.memory_summary())

    def is_model_loaded(self, name: str, mode: str):
        if (name == self._model_name and
            mode == self._model_mode):
            return True

        return False

    def unload_model(self) -> None:
        if getattr(self, '_model', None):
            del self._model

        gc.collect()
        torch.cuda.empty_cache()

        self._model_name = ''
        self._model_mode = ''

    def load_model(
        self,
        name: str,
        mode: str
    ) -> None:
        logging.info(f'loading model {name}...')
        self.unload_model()
        self._model = pipeline_for(
            name, mode, cache_dir=self.cache_dir)
        self._model_mode = mode
        self._model_name = name

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

                    # ?TODO, this is never caught, so why is it
                    # raised specially?
                    raise DGPUInferenceCancelled()

            return {}

        maybe_cancel_work(0)

        output_type = 'png'
        if 'output_type' in params:
            output_type = params['output_type']

        output = None
        output_hash = None
        try:
            name = params['model']

            match method:
                case 'diffuse' | 'txt2img' | 'img2img' | 'inpaint':
                    if not self.is_model_loaded(name, method):
                        self.load_model(name, method)

                    arguments = prepare_params_for_diffuse(
                        params, method, inputs)
                    prompt, guidance, step, seed, upscaler, extra_params = arguments

                    if 'flux' in name.lower():
                        extra_params['callback_on_step_end'] = maybe_cancel_work

                    else:
                        extra_params['callback'] = maybe_cancel_work
                        extra_params['callback_steps'] = 1

                    output = self._model(
                        prompt,
                        guidance_scale=guidance,
                        num_inference_steps=step,
                        generator=seed,
                        **extra_params
                    ).images[0]

                    output_binary = b''
                    match output_type:
                        case 'png':
                            if upscaler == 'x4':
                                input_img = output.convert('RGB')
                                up_img, _ = init_upscaler().enhance(
                                    convert_from_image_to_cv2(input_img), outscale=4)

                                output = convert_from_cv2_to_image(up_img)

                            output_binary = convert_from_img_to_bytes(output)

                        case _:
                            raise DGPUComputeError(f'Unsupported output type: {output_type}')

                    output_hash = sha256(output_binary).hexdigest()

                case 'upscale':
                    if self._model_mode != 'upscale':
                        self.unload_model()
                        self._model = init_upscaler()
                        self._model_mode = 'upscale'
                        self._model_name = 'realesrgan'

                    input_img = inputs[0].convert('RGB')
                    up_img, _ = self._model.enhance(
                        convert_from_image_to_cv2(input_img), outscale=4)

                    output = convert_from_cv2_to_image(up_img)

                    output_binary = convert_from_img_to_bytes(output)
                    output_hash = sha256(output_binary).hexdigest()

                case _:
                    raise DGPUComputeError('Unsupported compute method')

        except BaseException as err:
            logging.error(err)
            # to see the src exc in tb
            raise DGPUComputeError(str(err)) from err

        finally:
            torch.cuda.empty_cache()

        return output_hash, output
