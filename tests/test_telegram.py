#!/usr/bin/python

import trio
import trio_asyncio

from skynet_bot.brain import run_skynet
from skynet_bot.frontend import open_skynet_rpc
from skynet_bot.frontend.telegram import run_skynet_telegram


def test_run_tg_bot():
    async def main():
        async with trio.open_nursery() as n:
            await n.start(
                run_skynet,
                'skynet', '3GbZd6UojbD8V7zWpeFn', 'ancap.tech:34508')
            n.start_soon(
                run_skynet_telegram,
                '5853245787:AAFEmv3EjJ_qJ8d_vmOpi6o6HFHUf8a0uCQ')


    trio_asyncio.run(main)
