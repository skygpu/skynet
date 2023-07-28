#!/usr/bin/python

import logging
import traceback

from datetime import datetime, timedelta

import trio

from leap.hyperion import HyperionAPI

from . import IPFSHTTP


MAX_TIME = timedelta(seconds=20)


class SkynetPinner:

    def __init__(
        self,
        hyperion: HyperionAPI,
        ipfs_http: IPFSHTTP
    ):
        self.hyperion = hyperion
        self.ipfs_http = ipfs_http

        self._pinned = {}
        self._now = datetime.now()

    def is_pinned(self, cid: str):
        pin_time = self._pinned.get(cid)
        return pin_time

    def pin_cids(self, cids: list[str]):
        for cid in cids:
            self._pinned[cid] = self._now

    def cleanup_old_cids(self):
        cids = list(self._pinned.keys())
        for cid in cids:
            if (self._now - self._pinned[cid]) > MAX_TIME * 2:
                del self._pinned[cid]

    async def capture_enqueues(self, after: datetime):
        enqueues = await self.hyperion.aget_actions(
            account='telos.gpu',
            filter='telos.gpu:enqueue',
            sort='desc',
            after=after.isoformat(),
            limit=1000
        )

        logging.info(f'got {len(enqueues["actions"])} enqueue actions.')

        cids = []
        for action in enqueues['actions']:
            cid = action['act']['data']['binary_data']
            if cid and not self.is_pinned(cid):
                cids.append(cid)

        return cids

    async def capture_submits(self, after: datetime):
        submits = await self.hyperion.aget_actions(
            account='telos.gpu',
            filter='telos.gpu:submit',
            sort='desc',
            after=after.isoformat(),
            limit=1000
        )

        logging.info(f'got {len(submits["actions"])} submits actions.')

        cids = []
        for action in submits['actions']:
            cid = action['act']['data']['ipfs_hash']
            if cid and not self.is_pinned(cid):
                cids.append(cid)

        return cids

    async def task_pin(self, cid: str):
        logging.info(f'pinning {cid}...')
        for _ in range(6):
            try:
                with trio.move_on_after(5):
                    resp = await self.ipfs_http.a_pin(cid)
                    if resp.status_code != 200:
                        logging.error(f'error pinning {cid}:\n{resp.text}')
                        del self._pinned[cid]

                    else:
                        logging.info(f'pinned {cid}')
                        return

            except trio.TooSlowError:
                logging.error(f'timed out pinning {cid}')

        logging.error(f'gave up pinning {cid}')

    async def pin_forever(self):
        async with trio.open_nursery() as n:
            while True:
                try:
                    self._now = datetime.now()
                    self.cleanup_old_cids()

                    prev_second = self._now - MAX_TIME

                    cids = [
                        *(await self.capture_enqueues(prev_second)),
                        *(await self.capture_submits(prev_second))
                    ]

                    self.pin_cids(cids)

                    for cid in cids:
                        n.start_soon(self.task_pin, cid)

                except OSError as e:
                    traceback.print_exc()

                except KeyboardInterrupt:
                    break

                await trio.sleep(1)

