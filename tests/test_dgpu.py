#!/usr/bin/python

import time
import json
import logging

import trio
import pynng
import tractor
import trio_asyncio

from skynet_bot.gpu import open_gpu_worker
from skynet_bot.dgpu import open_dgpu_node
from skynet_bot.types import *
from skynet_bot.brain import run_skynet
from skynet_bot.constants import *
from skynet_bot.frontend import open_skynet_rpc, rpc_call


def test_dgpu_simple():
    async def main():
        async with trio.open_nursery() as n:
            await n.start(
                run_skynet,
                'skynet', '3GbZd6UojbD8V7zWpeFn', 'ancap.tech:34508')

            await trio.sleep(2)

            for i in range(3):
                n.start_soon(open_dgpu_node)

            await trio.sleep(1)
            start = time.time()
            async def request_img():
                with pynng.Req0(dial=DEFAULT_RPC_ADDR) as rpc_sock:
                    res = await rpc_call(
                        rpc_sock, 'tg+1', 'txt2img', {
                            'prompt': 'test',
                            'step': 28,
                            'width': 512, 'height': 512,
                            'guidance': 7.5,
                            'seed': None,
                            'algo': 'stable',
                            'upscaler': None
                        })

                    logging.info(res)

            async with trio.open_nursery() as inner_n:
                for i in range(3):
                    inner_n.start_soon(request_img)

            logging.info(f'time elapsed: {time.time() - start}')
            n.cancel_scope.cancel()


    trio_asyncio.run(main)
