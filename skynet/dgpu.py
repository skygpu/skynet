#!/usr/bin/python

import gc
import io
import json
import time
import random
import logging

from PIL import Image
from typing import List, Optional
from hashlib import sha256

import trio
import asks
import torch

from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, collect_stdout

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers.models import UNet2DConditionModel

from .ipfs import IPFSDocker, open_ipfs_node
from .utils import *
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
    account: str,
    permission: str,
    cleos: CLEOS,
    key: str = None,
    initial_algos: Optional[List[str]] = None
):

    logging.basicConfig(level=logging.INFO)
    logging.info(f'starting dgpu node!')
    logging.info(f'launching toolchain container!')

    if key:
        cleos.setup_wallet(key)

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

    def gpu_compute_one(method: str, params: dict, binext: Optional[bytes] = None):
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
                    if algo not in ALGOS:
                        raise DGPUComputeError(f'Unknown algo \"{algo}\"')

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
                        generator=torch.manual_seed(int(params['seed']))
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
                    img_sha = sha256(raw_img).hexdigest()
                    logging.info(f'final img size {len(raw_img)} bytes.')

                    logging.info(params)

                    return img_sha, raw_img

                except BaseException as e:
                    logging.error(e)
                    raise DGPUComputeError(str(e))

                finally:
                    torch.cuda.empty_cache()

            case _:
                raise DGPUComputeError('Unsupported compute method')

    async def get_work_requests_last_hour():
        logging.info('get_work_requests_last_hour')
        return await cleos.aget_table(
            'telos.gpu', 'telos.gpu', 'queue',
            index_position=2,
            key_type='i64',
            lower_bound=int(time.time()) - 3600
        )

    async def get_status_by_request_id(request_id: int):
        logging.info('get_status_by_request_id')
        return await cleos.aget_table(
            'telos.gpu', request_id, 'status')

    async def get_global_config():
        logging.info('get_global_config')
        return (await cleos.aget_table(
            'telos.gpu', 'telos.gpu', 'config'))[0]

    def get_worker_balance():
        logging.info('get_worker_balance')
        rows = cleos.get_table(
            'telos.gpu', 'telos.gpu', 'users',
            index_position=1,
            key_type='name',
            lower_bound=account,
            upper_bound=account
        )
        if len(rows) == 1:
            return rows[0]['balance']
        else:
            return None

    async def get_user_nonce(user: str):
        logging.info('get_user_nonce')
        return (await cleos.aget_table(
            'telos.gpu', 'telos.gpu', 'users',
            index_position=1,
            key_type='name',
            lower_bound=user,
            upper_bound=user
        ))[0]['nonce']

    def begin_work(request_id: int):
        logging.info('begin_work')
        ec, out = cleos.push_action(
            'telos.gpu',
            'workbegin',
            [account, request_id],
            f'{account}@{permission}'
        )
        assert ec == 0

    def cancel_work(request_id: int, reason: str):
        logging.info('cancel_work')
        ec, out = cleos.push_action(
            'telos.gpu',
            'workcancel',
            [account, request_id, reason],
            f'{account}@{permission}'
        )
        assert ec == 0

    def maybe_withdraw_all():
        logging.info('maybe_withdraw_all')
        balance = get_worker_balance()
        if not balance:
            return

        balance_amount = float(balance.split(' ')[0])
        if balance_amount > 0:
            ec, out = cleos.push_action(
                'telos.gpu',
                'withdraw',
                [account, balance],
                f'{account}@{permission}'
            )
            logging.info(collect_stdout(out))
            assert ec == 0

    async def find_my_results():
        logging.info('find_my_results')
        return await cleos.aget_table(
            'telos.gpu', 'telos.gpu', 'results',
            index_position=4,
            key_type='name',
            lower_bound=account,
            upper_bound=account
        )

    ipfs_node = None
    def publish_on_ipfs(img_sha: str, raw_img: bytes):
        logging.info('publish_on_ipfs')
        img = Image.open(io.BytesIO(raw_img))
        img.save(f'tmp/ipfs-docker-staging/image.png')

        ipfs_hash = ipfs_node.add('image.png')

        ipfs_node.pin(ipfs_hash)

        return ipfs_hash

    def submit_work(
        request_id: int,
        request_hash: str,
        result_hash: str,
        ipfs_hash: str
    ):
        logging.info('submit_work')
        ec, out = cleos.push_action(
            'telos.gpu',
            'submit',
            [account, request_id, request_hash, result_hash, ipfs_hash],
            f'{account}@{permission}'
        )

        print(collect_stdout(out))
        assert ec == 0

    config = await get_global_config()

    with open_ipfs_node() as ipfs_node:
        try:
            while True:
                maybe_withdraw_all()

                queue = await get_work_requests_last_hour()

                for req in queue:
                    rid = req['id']

                    my_results = [res['id'] for res in (await find_my_results())]
                    if rid in my_results:
                        continue

                    statuses = await get_status_by_request_id(rid)

                    if len(statuses) < config['verification_amount']:

                        # parse request
                        body = json.loads(req['body'])
                        binary = bytes.fromhex(req['binary_data'])
                        hash_str = (
                            str(await get_user_nonce(req['user']))
                            +
                            req['body']
                        )
                        logging.info(f'hashing: {hash_str}')
                        request_hash = sha256(hash_str.encode('utf-8')).hexdigest()

                        # TODO: validate request

                        # perform work
                        logging.info(f'working on {body}')

                        begin_work(rid)

                        try:
                            img_sha, raw_img = gpu_compute_one(
                                body['method'], body['params'], binext=binary)

                            ipfs_hash = publish_on_ipfs(img_sha, raw_img)

                            submit_work(rid, request_hash, img_sha, ipfs_hash)

                            break

                        except BaseException as e:
                            cancel_work(rid, str(e))

                    else:
                        logging.info(f'request {rid} already beign worked on, skip...')
                        continue

                await trio.sleep(1)

        except KeyboardInterrupt:
            ...



