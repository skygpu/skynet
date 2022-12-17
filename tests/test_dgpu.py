#!/usr/bin/python

import io
import time
import json
import base64
import logging

from hashlib import sha256
from functools import partial

import trio
import pytest
import tractor
import trio_asyncio

from PIL import Image

from skynet.brain import SkynetDGPUComputeError
from skynet.constants import *
from skynet.frontend import open_skynet_rpc


async def wait_for_dgpus(rpc, amount: int, timeout: float = 30.0):
    gpu_ready = False
    start_time = time.time()
    current_time = time.time()
    while not gpu_ready and (current_time - start_time) < timeout:
        res = await rpc('dgpu-test', 'dgpu_workers')
        logging.info(res)
        if res.result['ok'] >= amount:
            break

        await trio.sleep(1)
        current_time = time.time()

    assert (current_time - start_time) < timeout


_images = set()
async def check_request_img(
    i: int,
    width: int = 512,
    height: int = 512,
    expect_unique=True
):
    global _images

    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as rpc_call:
        res = await rpc_call(
            'tg+580213293', 'txt2img', {
                'prompt': 'red old tractor in a sunny wheat field',
                'step': 28,
                'width': width, 'height': height,
                'guidance': 7.5,
                'seed': None,
                'algo': list(ALGOS.keys())[i],
                'upscaler': None
            })

        if 'error' in res.result:
            raise SkynetDGPUComputeError(json.dumps(res.result))

        img_raw = base64.b64decode(bytes.fromhex(res.result['img']))
        img_sha = sha256(img_raw).hexdigest()
        img = Image.frombytes(
            'RGB', (width, height), img_raw)

        if expect_unique and img_sha in _images:
            raise ValueError('Duplicated image sha: {img_sha}')

        _images.add(img_sha)

        logging.info(f'img sha256: {img_sha} size: {len(img_raw)}')

        assert len(img_raw) > 100000


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker_compute_error(dgpu_workers):
    '''Attempt to generate a huge image and check we get the right error,
    then generate a smaller image to show gpu worker recovery
    '''

    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 1)

        with pytest.raises(SkynetDGPUComputeError) as e:
            await check_request_img(0, width=4096, height=4096)

        logging.info(e)

        await check_request_img(0)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj', 'stable'])], indirect=True)
async def test_dgpu_workers(dgpu_workers):
    '''Generate two images in a single dgpu worker using
    two different models.
    '''

    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 1)

        await check_request_img(0)
        await check_request_img(1)


@pytest.mark.parametrize(
    'dgpu_workers', [(2, ['midj'])], indirect=True)
async def test_dgpu_workers_two(dgpu_workers):
    '''Generate two images in two separate dgpu workers
    '''
    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 2)

        async with trio.open_nursery() as n:
            n.start_soon(check_request_img, 0)
            n.start_soon(check_request_img, 0)


@pytest.mark.parametrize(
    'dgpu_workers', [(1, ['midj'])], indirect=True)
async def test_dgpu_worker_algo_swap(dgpu_workers):
    '''Generate an image using a non default model
    '''
    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 1)
        await check_request_img(5)


@pytest.mark.parametrize(
    'dgpu_workers', [(3, ['midj'])], indirect=True)
async def test_dgpu_rotation_next_worker(dgpu_workers):
    '''Connect three dgpu workers, disconnect and check next_worker
    rotation happens correctly
    '''
    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 3)

        res = await test_rpc('testing-rpc', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        await check_request_img(0)

        res = await test_rpc('testing-rpc', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 1

        await check_request_img(0)

        res = await test_rpc('testing-rpc', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 2

        await check_request_img(0)

        res = await test_rpc('testing-rpc', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0


@pytest.mark.parametrize(
    'dgpu_workers', [(3, ['midj'])], indirect=True)
async def test_dgpu_rotation_next_worker_disconnect(dgpu_workers):
    '''Connect three dgpu workers, disconnect the first one and check
    next_worker rotation happens correctly
    '''
    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as test_rpc:
        await wait_for_dgpus(test_rpc, 3)

        await trio.sleep(3)

        # stop worker who's turn is next
        for _ in range(2):
            ec, out = dgpu_workers[0].exec_run(['pkill', '-INT', '-f', 'skynet'])
            assert ec == 0

        dgpu_workers[0].wait()

        res = await test_rpc('testing-rpc', 'dgpu_workers')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 2

        async with trio.open_nursery() as n:
            n.start_soon(check_request_img, 0)
            n.start_soon(check_request_img, 0)


async def test_dgpu_no_ack_node_disconnect(skynet_running):
    async with open_skynet_rpc(
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as rpc_call:

        res = await rpc_call('dgpu-0', 'dgpu_online')
        logging.info(res)
        assert 'ok' in res.result

        await wait_for_dgpus(rpc_call, 1)

        with pytest.raises(SkynetDGPUComputeError) as e:
            await check_request_img(0)

        assert 'dgpu failed to acknowledge request' in str(e)

        res = await rpc_call('testing-rpc', 'dgpu_workers')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0

