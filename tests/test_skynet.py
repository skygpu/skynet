#!/usr/bin/python

import logging

import trio
import pynng
import pytest
import trio_asyncio

from skynet.brain import run_skynet
from skynet.structs import *
from skynet.network import SessionServer
from skynet.frontend import open_skynet_rpc


async def test_skynet(skynet_running):
    ...


async def test_skynet_attempt_insecure(skynet_running):
    with pytest.raises(trio.TooSlowError) as e:
        with open_skynet_rpc('bad-actor') as session:
            with trio.fail_after(5):
                await session.rpc('skynet_shutdown')


async def test_skynet_dgpu_connection_simple(skynet_running):

    async def rpc_handler(req, ctx):
        ...

    fake_dgpu_addr = 'tcp://127.0.0.1:41001'
    rpc_server = SessionServer(
        fake_dgpu_addr,
        rpc_handler,
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    )

    with open_skynet_rpc(
        'dgpu-0',
        cert_name='whitelist/testing.cert',
        key_name='testing.key'
    ) as session:
        # check 0 nodes are connected
        res = await session.rpc('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == None

        async with rpc_server.open() as rpc_server:
            # connect 1 dgpu
            res = await session.rpc(
                'dgpu_online', {'dgpu_addr': fake_dgpu_addr})
            assert 'ok' in res.result

            # check 1 node is connected
            res = await session.rpc('dgpu_workers')
            assert 'ok' in res.result
            assert res.result['ok'] == 1

            # check next worker is 0
            res = await session.rpc('dgpu_next')
            assert 'ok' in res.result
            assert res.result['ok'] == 0

            # disconnect 1 dgpu
            res = await session.rpc('dgpu_offline')
            assert 'ok' in res.result

        # check 0 nodes are connected
        res = await session.rpc('dgpu_workers')
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await session.rpc('dgpu_next')
        assert 'ok' in res.result
        assert res.result['ok'] == None
