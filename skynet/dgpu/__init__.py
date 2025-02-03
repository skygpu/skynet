import trio

from hypercorn.config import Config
from hypercorn.trio import serve
from quart_trio import QuartTrio as Quart

from skynet.dgpu.compute import ModelMngr
from skynet.dgpu.daemon import WorkerDaemon
from skynet.dgpu.network import NetConnector


async def open_dgpu_node(config: dict) -> None:
    '''
    Open a top level "GPU mgmt daemon", keep the
    `WorkerDaemon._snap: dict[str, list|dict]` table
    and *maybe* serve a `hypercorn` web API.

    '''
    conn = NetConnector(config)
    mm = ModelMngr(config)
    daemon = WorkerDaemon(mm, conn, config)

    api: Quart|None = None
    if 'api_bind' in config:
        api_conf = Config()
        api_conf.bind = [config['api_bind']]
        api: Quart = await daemon.generate_api()

    tn: trio.Nursery
    async with trio.open_nursery() as tn:
        tn.start_soon(daemon.snap_updater_task)

        # TODO, consider a more explicit `as hypercorn_serve`
        # to clarify?
        if api:
            tn.start_soon(serve, api, api_conf)

        # block until cancelled
        await daemon.serve_forever()
