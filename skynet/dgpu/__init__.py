#!/usr/bin/python

from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.daemon import SkynetDGPUDaemon
from skynet.dgpu.network import SkynetGPUConnector


async def open_dgpu_node(config: dict):
    conn = SkynetGPUConnector(config)
    mm = SkynetMM(config)

    await SkynetDGPUDaemon(mm, conn, config).serve_forever()
