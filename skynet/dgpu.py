#!/usr/bin/python

import gc
import io
import json
import random
import logging

from PIL import Image
from typing import List, Optional

import trio
import torch

from pynng import Context
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers.models import UNet2DConditionModel

from .utils import *
from .network import *
from .protobuf import *
from .constants import *


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


class DGPUComputeError(BaseException):
    ...


async def open_dgpu_node(
    cert_name: str,
    unique_id: str,
    key_name: Optional[str],
    rpc_address: str = DEFAULT_RPC_ADDR,
    dgpu_address: str = DEFAULT_DGPU_ADDR,
    initial_algos: Optional[List[str]] = None
):
    logging.basicConfig(level=logging.DEBUG)
    logging.info(f'starting dgpu node!')
    logging.info(f'loading models...')

    upscaler = init_upscaler()
    initial_algos = (
        initial_algos
        if initial_algos else DEFAULT_INITAL_ALGOS
    )
    models = {}
    for algo in initial_algos:
        models[algo] = {
            'pipe': pipeline_for(algo),
            'generated': 0
        }
        logging.info(f'loaded {algo}.')

    logging.info('memory summary:')
    logging.info('\n' + torch.cuda.memory_summary())

    async def gpu_compute_one(method: str, params: dict, binext: Optional[bytes] = None):
        match method:
            case 'diffuse':
                image = None
                algo = params['algo']
                if binext:
                    algo += 'img'
                    image = Image.open(io.BytesIO(binext))
                    w, h = image.size
                    logging.info(f'user sent img of size {image.size}')

                    if w > 512 or h > 512:
                        image.thumbnail((512, 512))
                        logging.info(f'resized it to {image.size}')

                if algo not in models:
                    logging.info(f'{algo} not in loaded models, swapping...')
                    least_used = list(models.keys())[0]
                    for model in models:
                        if models[least_used]['generated'] > models[model]['generated']:
                            least_used = model

                    del models[least_used]
                    gc.collect()

                    models[algo] = {
                        'pipe': pipeline_for(params['algo'], image=True if binext else False),
                        'generated': 0
                    }
                    logging.info(f'swapping done.')

                _params = {}
                logging.info(method)
                logging.info(json.dumps(params, indent=4))
                logging.info(f'binext: {len(binext) if binext else 0} bytes')
                if binext:
                    _params['image'] = image
                    _params['strength'] = params['strength']

                else:
                    _params['width'] = int(params['width'])
                    _params['height'] = int(params['height'])

                try:
                    image = models[algo]['pipe'](
                        params['prompt'],
                        **_params,
                        guidance_scale=params['guidance'],
                        num_inference_steps=int(params['step']),
                        generator=torch.Generator("cuda").manual_seed(
                            int(params['seed']) if params['seed'] else random.randint(0, 2 ** 64)
                        )
                    ).images[0]

                    if params['upscaler'] == 'x4':
                        logging.info(f'size: {len(image.tobytes())}')
                        logging.info('performing upscale...')
                        input_img = image.convert('RGB')
                        up_img, _ = upscaler.enhance(
                            convert_from_image_to_cv2(input_img), outscale=4)

                        image = convert_from_cv2_to_image(up_img)
                        logging.info('done')

                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    raw_img = img_byte_arr.getvalue()
                    logging.info(f'final img size {len(raw_img)} bytes.')

                    return raw_img

                except BaseException as e:
                    logging.error(e)
                    raise DGPUComputeError(str(e))

                finally:
                    torch.cuda.empty_cache()

            case _:
                raise DGPUComputeError('Unsupported compute method')

    async def rpc_handler(req: SkynetRPCRequest, ctx: Context):
        result = {}
        resp = SkynetRPCResponse()

        match req.method:
            case 'dgpu_time':
                result = {'ok': time_ms()}

            case _:
                logging.debug(f'dgpu got one request: {req.method}')
                try:
                    resp.bin = await gpu_compute_one(
                        req.method, MessageToDict(req.params),
                        binext=req.bin if req.bin else None
                    )
                    logging.debug(f'dgpu processed one request')

                except DGPUComputeError as e:
                    result = {'error': str(e)}

        resp.result.update(result)
        return resp

    rpc_server = SessionServer(
        dgpu_address,
        rpc_handler,
        cert_name=cert_name,
        key_name=key_name
    )
    skynet_rpc = SessionClient(
        rpc_address,
        unique_id,
        cert_name=cert_name,
        key_name=key_name
    )
    skynet_rpc.connect()


    async with rpc_server.open() as rpc_server:
        res = await skynet_rpc.rpc(
            'dgpu_online', {
                'dgpu_addr': rpc_server.addr,
                'cert': cert_name
            })

        assert 'ok' in res.result

        try:
            await trio.sleep_forever()

        except KeyboardInterrupt:
            logging.info('interrupt caught, stopping...')

        finally:
            res = await skynet_rpc.rpc('dgpu_offline')
            assert 'ok' in res.result
