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


@tractor.context
async def open_fake_worker(
    ctx: tractor.Context,
    start_algo: str,
    mem_fraction: float
):
    log = tractor.log.get_logger(name='gpu', _root_name='skynet')
    log.info(f'starting gpu worker with algo {start_algo}...')
    current_algo = start_algo
    log.info('pipeline loaded')
    await ctx.started()
    async with ctx.open_stream() as bus:
        async for ireq in bus:
            if ireq:
                await bus.send('hello!')
            else:
                break

def test_gpu_worker():
    log = tractor.log.get_logger(name='root', _root_name='skynet')
    async def main():
        async with (
            tractor.open_nursery(debug_mode=True) as an,
            trio.open_nursery() as n
        ):
            portal = await an.start_actor(
                'gpu_worker',
                enable_modules=[__name__],
                debug_mode=True
            )

            log.info('portal opened')
            async with (
                portal.open_context(
                    open_fake_worker,
                    start_algo='midj',
                    mem_fraction=0.6
                ) as (ctx, _),
                ctx.open_stream() as stream,
            ):
                log.info('opened worker sending req...')
                ireq = ImageGenRequest(
                    prompt='a red tractor on a wheat field',
                    step=28,
                    width=512, height=512,
                    guidance=10, seed=None,
                    algo='midj', upscaler=None)

                await stream.send(ireq)
                log.info('sent, await respnse')
                async for msg in stream:
                    log.info(f'got {msg}')
                    break

                assert msg == 'hello!'
                await stream.send(None)
                log.info('done.')

            await portal.cancel_actor()

    trio.run(main)


def test_gpu_two_workers():
    async def main():
        outputs = []
        async with (
            tractor.open_actor_cluster(
                modules=[__name__],
                count=2,
                names=[0, 1]) as portal_map,
            tractor.trionics.gather_contexts((
                portal.open_context(
                    open_fake_worker,
                    start_algo='midj',
                    mem_fraction=0.333)
                for portal in portal_map.values()
            )) as contexts,
            trio.open_nursery() as n
        ):
            ireq = ImageGenRequest(
                prompt='a red tractor on a wheat field',
                step=28,
                width=512, height=512,
                guidance=10, seed=None,
                algo='midj', upscaler=None)

            async def get_img(i):
                ctx = contexts[i]
                async with ctx.open_stream() as stream:
                    await stream.send(ireq)
                    async for img in stream:
                        outputs[i] = img
                        await portal_map[i].cancel_actor()

            n.start_soon(get_img, 0)
            n.start_soon(get_img, 1)


        assert len(outputs) == 2

    trio.run(main)


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
