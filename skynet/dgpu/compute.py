'''
Skynet Memory Manager

'''

import gc
import logging

from hashlib import sha256
from contextlib import contextmanager as cm

import trio
import torch

from skynet.dgpu.tui import maybe_update_tui
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

_model_name: str = ''
_model_mode: str = ''
_model = None

@cm
def maybe_load_model(name: str, mode: str):
    if mode == 'diffuse':
        mode = 'txt2img'

    global _model_name, _model_mode, _model

    if _model_name != name or _model_mode != mode:
        # unload model
        _model = None
        gc.collect()
        torch.cuda.empty_cache()

        _model_name = _model_mode = ''

        # load model
        if mode == 'upscale':
            _model = init_upscaler()

        else:
            _model = pipeline_for(
                name, mode, cache_dir='hf_home')

        _model_name = name
        _model_mode = mode

        logging.debug('memory summary:')
        logging.debug('\n' + torch.cuda.memory_summary())

    yield


def compute_one(
    request_id: int,
    method: str,
    params: dict,
    inputs: list[bytes] = [],
    should_cancel = None
):
    if method == 'diffuse':
        method = 'txt2img'

    global _model, _model_name, _model_mode

    # validate correct model is loaded
    assert params['model'] == _model_name
    assert method == _model_mode

    total_steps = params['step']
    def inference_step_wakeup(*args, **kwargs):
        '''This is a callback function that gets invoked every inference step,
        we need to raise an exception here if we need to cancel work
        '''
        step = args[0]
        # compat with callback_on_step_end
        if not isinstance(step, int):
            step = args[1]

        maybe_update_tui(lambda tui: tui.set_progress(step, done=total_steps))

        if should_cancel:
            should_raise = trio.from_thread.run(should_cancel, request_id)

        if should_raise:
            logging.warning(f'CANCELLING work at step {step}')
            raise DGPUInferenceCancelled('network cancel')

        return {}

    maybe_update_tui(lambda tui: tui.set_status(f'Request #{request_id}'))

    inference_step_wakeup(0)

    output_type = 'png'
    if 'output_type' in params:
        output_type = params['output_type']

    output = None
    output_hash = None
    try:
        name = params['model']

        match method:
            case 'txt2img' | 'img2img' | 'inpaint':
                arguments = prepare_params_for_diffuse(
                    params, method, inputs)
                prompt, guidance, step, seed, upscaler, extra_params = arguments

                if 'flux' in name.lower():
                    extra_params['callback_on_step_end'] = inference_step_wakeup

                else:
                    extra_params['callback'] = inference_step_wakeup
                    extra_params['callback_steps'] = 1

                output = _model(
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
                input_img = inputs[0].convert('RGB')
                up_img, _ = _model.enhance(
                    convert_from_image_to_cv2(input_img), outscale=4)

                output = convert_from_cv2_to_image(up_img)

                output_binary = convert_from_img_to_bytes(output)
                output_hash = sha256(output_binary).hexdigest()

            case _:
                raise DGPUComputeError('Unsupported compute method')

    except BaseException as err:
        raise DGPUComputeError(str(err)) from err

    maybe_update_tui(lambda tui: tui.set_status(''))

    return output_hash, output
