#!/usr/bin/python

import logging

import trio
import trio_asyncio

from skynet_bot.types import *
from skynet_bot.brain import run_skynet
from skynet_bot.frontend import open_skynet_rpc


async def test_skynet_dgpu_connection_simple(skynet_running):
    async with open_skynet_rpc() as rpc_call:
        # check 0 nodes are connected
        res = await rpc_call('dgpu-0', 'dgpu_workers')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await rpc_call('dgpu-0', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == None

        # connect 1 dgpu
        res = await rpc_call(
            'dgpu-0', 'dgpu_online', {'max_tasks': 3})
        logging.info(res)
        assert 'ok' in res.result

        # check 1 node is connected
        res = await rpc_call('dgpu-0', 'dgpu_workers')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 1

        # check next worker is 0
        res = await rpc_call('dgpu-0', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # disconnect 1 dgpu
        res = await rpc_call(
            'dgpu-0', 'dgpu_offline')
        logging.info(res)
        assert 'ok' in res.result

        # check 0 nodes are connected
        res = await rpc_call('dgpu-0', 'dgpu_workers')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == 0

        # check next worker is None
        res = await rpc_call('dgpu-0', 'dgpu_next')
        logging.info(res)
        assert 'ok' in res.result
        assert res.result['ok'] == None
