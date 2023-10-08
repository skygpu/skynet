#!/usr/bin/python

import trio

from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.daemon import SkynetDGPUDaemon
from skynet.dgpu.network import SkynetGPUConnector


async def open_dgpu_node(config: dict):
    conn = SkynetGPUConnector(config)
    mm = SkynetMM(config)
    daemon = SkynetDGPUDaemon(mm, conn, config)

    async with trio.open_nursery() as n:
        n.start_soon(daemon.snap_updater_task)
        await daemon.serve_forever()
