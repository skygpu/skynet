import logging
from contextlib import asynccontextmanager as acm

import trio
import urwid

from skynet.config import Config
from skynet.dgpu.tui import init_tui
from skynet.dgpu.daemon import dgpu_serve_forever
from skynet.dgpu.network import NetConnector


@acm
async def open_worker(config: Config):
    # suppress logs from httpx (logs url + status after every query)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tui = None
    if config.tui:
        tui = init_tui(config)

    conn = NetConnector(config)

    try:
        n: trio.Nursery
        async with trio.open_nursery() as n:
            if tui:
                n.start_soon(tui.run)

            yield conn

    except *urwid.ExitMainLoop:
        ...


async def _dgpu_main(config: Config):
    async with open_worker(config) as conn:
        await dgpu_serve_forever(config, conn)
