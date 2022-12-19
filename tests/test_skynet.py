#!/usr/bin/python

import logging

import trio
import pynng
import pytest
import trio_asyncio

from skynet.brain import run_skynet
from skynet.structs import *
from skynet.frontend import open_skynet_rpc


async def test_skynet(skynet_running):
    ...


async def test_skynet_attempt_insecure(skynet_running):
    with pytest.raises(pynng.exceptions.NNGException) as e:
        async with open_skynet_rpc('bad-actor'):
           ...

    assert str(e.value) == 'Connection shutdown'


async def test_skynet_dgpu_connection_simple(skynet_running):
    async with open_skynet_rpc(
        'dgpu-0',
        security=True,
        cert_name='whitelist/testing',
        key_name='testing'
    ) as rpc_call:
        # check 0 nodes are connected
        res = await rpc_call('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await rpc_call('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == None

        # connect 1 dgpu
        res = await rpc_call('dgpu_online')
        assert 'ok' in res.result

        # check 1 node is connected
        res = await rpc_call('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 1

        # check next worker is 0
        res = await rpc_call('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # disconnect 1 dgpu
        res = await rpc_call('dgpu_offline')
        assert 'ok' in res.result

        # check 0 nodes are connected
        res = await rpc_call('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await rpc_call('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == None
