import logging

import trio
import urwid

from hypercorn.config import Config as HCConfig
from hypercorn.trio import serve
from quart_trio import QuartTrio as Quart

from skynet.config import Config
from skynet.dgpu.tui import init_tui
from skynet.dgpu.daemon import WorkerDaemon
from skynet.dgpu.network import NetConnector


async def open_dgpu_node(config: Config) -> None:
    '''
    Open a top level "GPU mgmt daemon", keep the
    `WorkerDaemon._snap: dict[str, list|dict]` table
    and *maybe* serve a `hypercorn` web API.

    '''
    # suppress logs from httpx (logs url + status after every query)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tui = None
    if config.tui:
        tui = init_tui()

    conn = NetConnector(config)
    daemon = WorkerDaemon(conn, config)

    api: Quart|None = None
    if config.api_bind:
        api_conf = HCConfig()
        api_conf.bind = [config.api_bind]
        api: Quart = await daemon.generate_api()

    tn: trio.Nursery
    async with trio.open_nursery() as tn:
        tn.start_soon(daemon.snap_updater_task)
        if tui:
            tn.start_soon(tui.run)

        # TODO, consider a more explicit `as hypercorn_serve`
        # to clarify?
        if api:
            logging.info(f'serving api @ {config["api_bind"]}')
            tn.start_soon(serve, api, api_conf)

        try:
            # block until cancelled
            await daemon.serve_forever()

        except *urwid.ExitMainLoop:
            ...
