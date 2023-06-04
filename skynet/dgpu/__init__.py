#!/usr/bin/python

import trio

from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.daemon import SkynetDGPUDaemon
from skynet.dgpu.network import SkynetGPUConnector


async def open_dgpu_node(config: dict):
    conn = SkynetGPUConnector(config)
    mm = SkynetMM(config)

    async with conn.open() as conn:
        await (SkynetDGPUDaemon(mm, conn, config)
            .serve_forever())
