#!/usr/bin/python

import trio

from hypercorn.config import Config
from hypercorn.trio import serve

from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.daemon import SkynetDGPUDaemon
from skynet.dgpu.network import SkynetGPUConnector


async def open_dgpu_node(config: dict):
    conn = SkynetGPUConnector(config)
    mm = SkynetMM(config)
    daemon = SkynetDGPUDaemon(mm, conn, config)

    api = None
    if 'api_bind' in config:
        api_conf = Config()
        api_conf.bind = [config['api_bind']]
        api = await daemon.generate_api()

    async with trio.open_nursery() as n:
        n.start_soon(daemon.snap_updater_task)

        if api:
            n.start_soon(serve, api, api_conf)

        await daemon.serve_forever()
