import logging
import warnings

import trio

from hypercorn.config import Config
from hypercorn.trio import serve
from quart_trio import QuartTrio as Quart

from skynet.dgpu.tui import WorkerMonitor
from skynet.dgpu.compute import ModelMngr
from skynet.dgpu.daemon import WorkerDaemon
from skynet.dgpu.network import NetConnector


def setup_logging_for_tui(level):
    warnings.filterwarnings("ignore")

    logger = logging.getLogger()
    logger.setLevel(level)

    fh = logging.FileHandler('dgpu.log')
    fh.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


async def open_dgpu_node(config: dict) -> None:
    '''
    Open a top level "GPU mgmt daemon", keep the
    `WorkerDaemon._snap: dict[str, list|dict]` table
    and *maybe* serve a `hypercorn` web API.

    '''
    # suppress logs from httpx (logs url + status after every query)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tui = None
    if config['tui']:
        setup_logging_for_tui(logging.INFO)
        tui = WorkerMonitor()

    conn = NetConnector(config, tui=tui)
    mm = ModelMngr(config, tui=tui)
    daemon = WorkerDaemon(mm, conn, config, tui=tui)

    api: Quart|None = None
    if 'api_bind' in config:
        api_conf = Config()
        api_conf.bind = [config['api_bind']]
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

        except *urwid.ExitMainLoop in ex_group:
            ...
