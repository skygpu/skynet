import logging

import trio
import urwid

from skynet.config import Config
from skynet.dgpu.tui import init_tui
from skynet.dgpu.daemon import serve_forever
from skynet.dgpu.network import NetConnector


async def _dgpu_main(config: Config) -> None:
    # suppress logs from httpx (logs url + status after every query)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tui = None
    if config.tui:
        tui = init_tui()

    conn = NetConnector(config)

    try:
        n: trio.Nursery
        async with trio.open_nursery() as n:
            if tui:
                n.start_soon(tui.run)

            await serve_forever(config, conn)

    except *urwid.ExitMainLoop:
        ...
