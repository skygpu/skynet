#!/usr/bin/python

import gc
import io
import trio
import json
import uuid
import time
import zlib
import random
import logging
import traceback

from PIL import Image
from typing import List, Optional
from pathlib import Path
from contextlib import ExitStack

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
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers.models import UNet2DConditionModel

from .utils import (
    pipeline_for,
    convert_from_cv2_to_image, convert_from_image_to_cv2
)
from .protobuf import *
from .frontend import open_skynet_rpc
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


class ReconnectingBus:

    def __init__(self, address: str, tls_config: Optional[TLSConfig]):
        self.address = address
        self.tls_config = tls_config

        self._stack = ExitStack()
        self._sock = None
        self._closed = True

    def connect(self):
        self._sock = self._stack.enter_context(
            pynng.Bus0(recv_max_size=0))
        self._sock.tls_config = self.tls_config
        self._sock.dial(self.address)
        self._closed = False

    async def arecv(self):
        while True:
            try:
                return await self._sock.arecv()

            except pynng.exceptions.Closed:
                if self._closed:
                    raise

    async def asend(self, msg):
        while True:
            try:
                return await self._sock.asend(msg)

            except pynng.exceptions.Closed:
                if self._closed:
                    raise

    def close(self):
        self._stack.close()
        self._stack = ExitStack()
        self._closed = True

    def reconnect(self):
        self.close()
        self.connect()


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

    async def gpu_compute_one(ireq: DiffusionParameters, image=None):
        algo = ireq.algo + 'img' if image else ireq.algo
        if algo not in models:
            least_used = list(models.keys())[0]
            for model in models:
                if models[least_used]['generated'] > models[model]['generated']:
                    least_used = model

            del models[least_used]
            gc.collect()

            models[algo] = {
                'pipe': pipeline_for(ireq.algo, image=True if image else False),
                'generated': 0
            }

        _params = {}
        if ireq.image:
            _params['image'] = image

        else:
            _params['width'] = int(ireq.width)
            _params['height'] = int(ireq.height)

        try:
            image = models[algo]['pipe'](
                ireq.prompt,
                **_params,
                guidance_scale=ireq.guidance,
                num_inference_steps=int(ireq.step),
                generator=torch.Generator("cuda").manual_seed(ireq.seed)
            ).images[0]

            if ireq.upscaler == 'x4':
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


    async with (
        open_skynet_rpc(
            unique_id,
            rpc_address=rpc_address,
            security=security,
            cert_name=cert_name,
            key_name=key_name
        ) as rpc_call,
        trio.open_nursery() as n
    ):

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

        dgpu_bus = ReconnectingBus(dgpu_address, tls_config)
        dgpu_bus.connect()

        last_msg = time.time()
        async def connection_refresher(refresh_time: int = 120):
            nonlocal last_msg
            while True:
                now = time.time()
                last_msg_time_delta = now - last_msg
                logging.info(f'time since last msg: {last_msg_time_delta}')
                if last_msg_time_delta > refresh_time:
                    dgpu_bus.reconnect()
                    logging.info('reconnected!')
                    last_msg = now

                await trio.sleep(refresh_time)

        n.start_soon(connection_refresher)

        res = await rpc_call('dgpu_online')
        assert 'ok' in res.result

        try:
            while True:
                msg = await dgpu_bus.arecv()

                img = None
                if b'BINEXT' in msg:
                    header, msg, img_raw = msg.split(b'%$%$')
                    logging.info(f'got img attachment of size {len(img_raw)}')
                    logging.info(img_raw[:10])
                    raw_img = zlib.decompress(img_raw)
                    logging.info(raw_img[:10])
                    img = Image.open(io.BytesIO(raw_img))
                    w, h = img.size
                    logging.info(f'user sent img of size {img.size}')

                    if w > 512 or h > 512:
                        img.thumbnail((512, 512))
                        logging.info(f'resized it to {img.size}')


                req = DGPUBusMessage()
                req.ParseFromString(msg)
                last_msg = time.time()

                if req.method == 'heartbeat':
                    rep = DGPUBusMessage(
                        rid=req.rid,
                        nid=unique_id,
                        method=req.method
                    )
                    rep.params.update({'time': int(time.time() * 1000)})

                    if security:
                        rep.auth.cert = cert_name
                        rep.auth.sig = sign_protobuf_msg(rep, tls_key)

                    await dgpu_bus.asend(rep.SerializeToString())
                    logging.info('heartbeat reply')
                    continue

                if req.nid != unique_id:
                    logging.info(
                        f'witnessed msg {req.rid}, node involved: {req.nid}')
                    continue

                if security:
                    verify_protobuf_msg(req, skynet_cert)


                ack_resp = DGPUBusMessage(
                    rid=req.rid,
                    nid=req.nid
                )
                ack_resp.params.update({'ack': {}})

                if security:
                    ack_resp.auth.cert = cert_name
                    ack_resp.auth.sig = sign_protobuf_msg(ack_resp, tls_key)

                # send ack
                await dgpu_bus.asend(ack_resp.SerializeToString())

                logging.info(f'sent ack, processing {req.rid}...')

                try:
                    img_req = DiffusionParameters(**req.params)

                    if not img_req.seed:
                        img_req.seed = random.randint(0, 2 ** 64)

                    img = await gpu_compute_one(img_req, image=img)
                    img_resp = DGPUBusMessage(
                        rid=req.rid,
                        nid=req.nid,
                        method='binary-reply'
                    )
                    img_resp.params.update({
                        'len': len(img),
                        'meta': img_req.to_dict()
                    })

                except DGPUComputeError as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    img_resp = DGPUBusMessage(
                        rid=req.rid,
                        nid=req.nid
                    )
                    img_resp.params.update({'error': str(e)})


                if security:
                    img_resp.auth.cert = cert_name
                    img_resp.auth.sig = sign_protobuf_msg(img_resp, tls_key)

                # send final image
                logging.info('sending img back...')
                raw_msg = img_resp.SerializeToString()
                await dgpu_bus.asend(raw_msg)
                logging.info(f'sent {len(raw_msg)} bytes.')
                if img_resp.method == 'binary-reply':
                    await dgpu_bus.asend(zlib.compress(img))
                    logging.info(f'sent {len(img)} bytes.')

        except KeyboardInterrupt:
            logging.info('interrupt caught, stopping...')
            n.cancel_scope.cancel()
            dgpu_bus.close()

        finally:
            res = await rpc_call('dgpu_offline')
            assert 'ok' in res.result
