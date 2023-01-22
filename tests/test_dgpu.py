#!/usr/bin/python

import io
import time
import json
import zlib
import logging

from typing import Optional
from hashlib import sha256
from functools import partial

import trio
import pytest

from PIL import Image
from google.protobuf.json_format import MessageToDict

from skynet.brain import SkynetDGPUComputeError
from skynet.network import get_random_port, SessionServer
from skynet.protobuf import SkynetRPCResponse
from skynet.frontend import open_skynet_rpc
from skynet.constants import *


async def wait_for_dgpus(session, amount: int, timeout: float = 30.0):
    gpu_ready = False
    with trio.fail_after(timeout):
        while not gpu_ready:
            res = await session.rpc('dgpu_workers')
            if res.result['ok'] >= amount:
                break

            await trio.sleep(1)


_images = set()
async def check_request_img(
    i: int,
    uid: str = '1',
    width: int = 512,
    height: int = 512,
    expect_unique = True,
    upscaler: Optional[str] = None
):
    global _images

    with open_skynet_rpc(
        uid,
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        res = await session.rpc(
            'dgpu_call', {
                'method': 'diffuse',
                'params': {
                    'prompt': 'red old tractor in a sunny wheat field',
                    'step': 28,
                    'width': width, 'height': height,
                    'guidance': 7.5,
                    'seed': None,
                    'algo': list(ALGOS.keys())[i],
                    'upscaler': upscaler
                }
            },
            timeout=60
        )

        if 'error' in res.result:
            raise SkynetDGPUComputeError(MessageToDict(res.result))

        img_raw = res.bin
        img_sha = sha256(img_raw).hexdigest()
        img = Image.open(io.BytesIO(img_raw))

        if expect_unique and img_sha in _images:
            raise ValueError('Duplicated image sha: {img_sha}')

        _images.add(img_sha)

        logging.info(f'img sha256: {img_sha} size: {len(img_raw)}')

        assert len(img_raw) > 100000

        return img


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker_compute_error(dgpu_workers):
    '''Attempt to generate a huge image and check we get the right error,
    then generate a smaller image to show gpu worker recovery
    '''

    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        with pytest.raises(SkynetDGPUComputeError) as e:
            await check_request_img(0, width=4096, height=4096)

        logging.info(e)

        await check_request_img(0)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker(dgpu_workers):
    '''Generate one image in a single dgpu worker
    '''

    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        await check_request_img(0)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj', 'stable'])], indirect=True)
async def test_dgpu_worker_two_models(dgpu_workers):
    '''Generate two images in a single dgpu worker using
    two different models.
    '''

    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        await check_request_img(0)
        await check_request_img(1)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker_upscale(dgpu_workers):
    '''Generate two images in a single dgpu worker using
    two different models.
    '''

    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        img = await check_request_img(0, upscaler='x4')

        assert img.size == (2048, 2048)


@pytest.mark.parametrize(
    'dgpu_workers', [(2, ['midj'])], indirect=True)
async def test_dgpu_workers_two(dgpu_workers):
    '''Generate two images in two separate dgpu workers
    '''
    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 2, timeout=60)

        async with trio.open_nursery() as n:
            n.start_soon(check_request_img, 0)
            n.start_soon(check_request_img, 0)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker_algo_swap(dgpu_workers):
    '''Generate an image using a non default model
    '''
    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)
        await check_request_img(5)


@pytest.mark.parametrize(
    'dgpu_workers', [(3, ['midj'])], indirect=True)
async def test_dgpu_rotation_next_worker(dgpu_workers):
    '''Connect three dgpu workers, disconnect and check next_worker
    rotation happens correctly
    '''
    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 3)

        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        await check_request_img(0)

        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == 1

        await check_request_img(0)

        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == 2

        await check_request_img(0)

        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == 0


@pytest.mark.parametrize(
    'dgpu_workers', [(3, ['midj'])], indirect=True)
async def test_dgpu_rotation_next_worker_disconnect(dgpu_workers):
    '''Connect three dgpu workers, disconnect the first one and check
    next_worker rotation happens correctly
    '''
    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 3)

        await trio.sleep(3)

        # stop worker who's turn is next
        for _ in range(2):
            ec, out = dgpu_workers[0].exec_run(['pkill', '-INT', '-f', 'skynet'])
            assert ec == 0

        dgpu_workers[0].wait()

        res = await session.rpc('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 2

        async with trio.open_nursery() as n:
            n.start_soon(check_request_img, 0)
            n.start_soon(check_request_img, 0)


async def test_dgpu_no_ack_node_disconnect(skynet_running):
    '''Mock a node that connects, gets a request but fails to
    acknowledge it, then check skynet correctly drops the node
    '''

    async def mock_rpc(req, ctx):
        resp = SkynetRPCResponse()
        resp.result.update({'error': 'can\'t do it mate'})
        return resp

    dgpu_addr = f'tcp://127.0.0.1:{get_random_port()}'
    mock_server = SessionServer(
        dgpu_addr,
        mock_rpc,
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    )

    async with mock_server.open():
        with open_skynet_rpc(
            'test-ctx',
            cert_name='whitelist/testing.cert',
            key_name='testing.key'
        ) as session:

            res = await session.rpc('dgpu_online', {
                'dgpu_addr': dgpu_addr,
                'cert': 'whitelist/testing.cert'
            })
            assert 'ok' in res.result

            await wait_for_dgpus(session, 1)

            with pytest.raises(SkynetDGPUComputeError) as e:
                await check_request_img(0)

            assert 'can\'t do it mate' in str(e.value)

            res = await session.rpc('dgpu_workers')
            assert 'ok' in res.result
            assert res.result['ok'] == 0


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_timeout_while_processing(dgpu_workers):
    '''Stop node while processing request to cause timeout and
    then check skynet correctly drops the node.
    '''
    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        async def check_request_img_raises():
            with pytest.raises(SkynetDGPUComputeError) as e:
                await check_request_img(0)

            assert 'timeout while processing request' in str(e)

        async with trio.open_nursery() as n:
            n.start_soon(check_request_img_raises)
            await trio.sleep(1)
            ec, out = dgpu_workers[0].exec_run(
                ['pkill', '-TERM', '-f', 'skynet'])
            assert ec == 0


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_img2img(dgpu_workers):

    with open_skynet_rpc(
        'test-ctx',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        await wait_for_dgpus(session, 1)

        await trio.sleep(2)

        res = await session.rpc(
            'dgpu_call', {
                'method': 'diffuse',
                'params': {
                    'prompt': 'red old tractor in a sunny wheat field',
                    'step': 28,
                    'width': 512, 'height': 512,
                    'guidance': 7.5,
                    'seed': None,
                    'algo': list(ALGOS.keys())[0],
                    'upscaler': None
                }
            },
            timeout=60
        )

        if 'error' in res.result:
            raise SkynetDGPUComputeError(MessageToDict(res.result))

        img_raw = res.bin
        img = Image.open(io.BytesIO(img_raw))
        img.save('txt2img.png')

        res = await session.rpc(
            'dgpu_call', {
                'method': 'diffuse',
                'params': {
                    'prompt': 'red ferrari in a sunny wheat field',
                    'step': 28,
                    'guidance': 8,
                    'strength': 0.7,
                    'seed': None,
                    'algo': list(ALGOS.keys())[0],
                    'upscaler': 'x4'
                }
            },
            binext=img_raw,
            timeout=60
        )

        if 'error' in res.result:
            raise SkynetDGPUComputeError(MessageToDict(res.result))

        img_raw = res.bin
        img = Image.open(io.BytesIO(img_raw))
        img.save('img2img.png')
