import logging
from contextlib import asynccontextmanager as acm

import trio
import urwid

from skynet.config import Config
from skynet.dgpu.tui import init_tui, WorkerMonitor
from skynet.dgpu.daemon import dgpu_serve_forever
from skynet.dgpu.network import NetConnector, maybe_open_contract_state_mngr


@acm
async def open_worker(config: Config):
    # suppress logs from httpx (logs url + status after every query)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tui: WorkerMonitor | None = None
    if config.tui:
        tui = init_tui(config)

    conn = NetConnector(config)
    try:
        async with maybe_open_contract_state_mngr(conn) as state_mngr:
            n: trio.Nursery
            async with trio.open_nursery() as n:
                if tui:
                    n.start_soon(tui.run)

                n.start_soon(dgpu_serve_forever, config, conn, state_mngr)

                yield conn, state_mngr

                n.cancel_scope.cancel()

    except *urwid.ExitMainLoop:
        ...


async def _dgpu_main(config: Config):
    async with open_worker(config):
        await trio.sleep_forever()
