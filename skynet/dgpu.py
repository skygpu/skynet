#!/usr/bin/python

import gc
import io
import trio
import json
import uuid
import base64
import random
import logging

from typing import List, Optional
from pathlib import Path
from contextlib import AsyncExitStack

import pynng
import torch

from pynng import TLSConfig
from OpenSSL.crypto import (
    load_privatekey,
    load_certificate,
    FILETYPE_PEM
)
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler
)
from diffusers.models import UNet2DConditionModel

from .structs import *
from .constants import *
from .frontend import open_skynet_rpc


def pipeline_for(algo: str, mem_fraction: float = 1.0):
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

    pipe = StableDiffusionPipeline.from_pretrained(
        ALGOS[algo], **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    pipe.enable_vae_slicing()

    return pipe.to("cuda")


class DGPUComputeError(BaseException):
    ...


async def open_dgpu_node(
    cert_name: str,
    unique_id: str,
    key_name: Optional[str],
    rpc_address: str = DEFAULT_RPC_ADDR,
    dgpu_address: str = DEFAULT_DGPU_ADDR,
    initial_algos: Optional[List[str]] = None,
    security: bool = True
):
    logging.basicConfig(level=logging.INFO)
    logging.info(f'starting dgpu node!')

    name = uuid.uuid4()

    logging.info(f'loading models...')

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

    logging.info('memory summary:\n')
    logging.info(torch.cuda.memory_summary())

    async def gpu_compute_one(ireq: ImageGenRequest):
        if ireq.algo not in models:
            least_used = list(models.keys())[0]
            for model in models:
                if models[least_used]['generated'] > models[model]['generated']:
                    least_used = model

            del models[least_used]
            gc.collect()

            models[ireq.algo] = {
                'pipe': pipeline_for(ireq.algo),
                'generated': 0
            }

        seed = ireq.seed if ireq.seed else random.randint(0, 2 ** 64)
        try:
            image = models[ireq.algo]['pipe'](
                ireq.prompt,
                width=ireq.width,
                height=ireq.height,
                guidance_scale=ireq.guidance,
                num_inference_steps=ireq.step,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images[0]
            return image.tobytes()

        except BaseException as e:
            logging.error(e)
            raise DGPUComputeError(str(e))

        finally:
            torch.cuda.empty_cache()


    async with open_skynet_rpc(
        unique_id,
        security=security,
        cert_name=cert_name,
        key_name=key_name
    ) as rpc_call:

        tls_config = None
        if security:
            # load tls certs
            if not key_name:
                key_name = cert_name

            certs_dir = Path(DEFAULT_CERTS_DIR).resolve()

            skynet_cert_path = certs_dir / 'brain.cert'
            tls_cert_path = certs_dir / f'{cert_name}.cert'
            tls_key_path = certs_dir / f'{key_name}.key'

            cert_name = tls_cert_path.stem

            skynet_cert_data = skynet_cert_path.read_text()
            skynet_cert = load_certificate(FILETYPE_PEM, skynet_cert_data)

            tls_cert_data = tls_cert_path.read_text()

            tls_key_data = tls_key_path.read_text()
            tls_key = load_privatekey(FILETYPE_PEM, tls_key_data)

            logging.info(f'skynet cert: {skynet_cert_path}')
            logging.info(f'dgpu cert:  {tls_cert_path}')
            logging.info(f'dgpu key: {tls_key_path}')

            dgpu_address = 'tls+' + dgpu_address
            tls_config = TLSConfig(
                TLSConfig.MODE_CLIENT,
                own_key_string=tls_key_data,
                own_cert_string=tls_cert_data,
                ca_string=skynet_cert_data)

        logging.info(f'connecting to {dgpu_address}')
        with pynng.Bus0() as dgpu_sock:
            dgpu_sock.tls_config = tls_config
            dgpu_sock.dial(dgpu_address)

            res = await rpc_call('dgpu_online')
            assert 'ok' in res.result

            try:
                while True:
                    msg = await dgpu_sock.arecv()
                    req = DGPUBusRequest(
                        **json.loads(msg.decode()))

                    if req.nid != unique_id:
                        logging.info(
                            f'witnessed msg {req.rid}, node involved: {req.nid}')
                        continue

                    if security:
                        req.verify(skynet_cert)

                    ack_resp = DGPUBusResponse(
                        rid=req.rid,
                        nid=req.nid,
                        params={'ack': {}}
                    )

                    if security:
                        ack_resp.sign(tls_key, cert_name)

                    # send ack
                    await dgpu_sock.asend(
                        json.dumps(ack_resp.to_dict()).encode())

                    logging.info(f'sent ack, processing {req.rid}...')

                    try:
                        img = await gpu_compute_one(
                            ImageGenRequest(**req.params))
                        img_resp = DGPUBusResponse(
                            rid=req.rid,
                            nid=req.nid,
                            params={'img': base64.b64encode(img).hex()}
                        )

                    except DGPUComputeError as e:
                        img_resp = DGPUBusResponse(
                            rid=req.rid,
                            nid=req.nid,
                            params={'error': str(e)}
                        )

                    if security:
                        img_resp.sign(tls_key, cert_name)

                    # send final image
                    await dgpu_sock.asend(
                        json.dumps(img_resp.to_dict()).encode())

            except KeyboardInterrupt:
                logging.info('interrupt caught, stopping...')

            finally:
                res = await rpc_call('dgpu_offline')
                assert 'ok' in res.result