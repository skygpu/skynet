#!/usr/bin/python

import json
import random
import logging
import traceback

from hashlib import sha256
from functools import partial

import trio
import tractor

from skynet.dgpu.errors import *
from skynet.dgpu.compute import SkynetMM, _tractor_static_compute_one
from skynet.dgpu.network import SkynetGPUConnector


def convert_reward_to_int(reward_str):
    int_part, decimal_part = (
        reward_str.split('.')[0],
        reward_str.split('.')[1].split(' ')[0]
    )
    return int(int_part + decimal_part)


class SkynetDGPUDaemon:

    def __init__(
        self,
        mm: SkynetMM,
        conn: SkynetGPUConnector,
        config: dict
    ):
        self.mm = mm
        self.conn = conn
        self.auto_withdraw = (
            config['auto_withdraw']
            if 'auto_withdraw' in config else False
        )

        self.non_compete = set()
        if 'non_compete' in config:
            self.non_compete = set(config['non_compete'])

        self.model_whitelist = set()
        if 'model_whitelist' in config:
            self.model_whitelist = set(config['model_whitelist'])

        self.model_blacklist = set()
        if 'model_blacklist' in config:
            self.model_blacklist = set(config['model_blacklist'])

        self.backend = 'sync-on-thread'
        if 'backend' in config:
            self.backend = config['backend']

        self._snap = {
            'queue': [],
            'requests': {},
            'my_results': []
        }

    async def should_cancel_work(self, request_id: int):
        competitors = set([
            status['worker']
            for status in self._snap['requests'][request_id]
            if status['worker'] != self.conn.account
        ])
        return bool(self.non_compete & competitors)


    async def snap_updater_task(self):
        while True:
            self._snap = await self.conn.get_full_queue_snapshot()
            await trio.sleep(1)

    async def serve_forever(self):
        try:
            while True:
                if self.auto_withdraw:
                    await self.conn.maybe_withdraw_all()

                queue = self._snap['queue']

                random.shuffle(queue)
                queue = sorted(
                    queue,
                    key=lambda req: convert_reward_to_int(req['reward']),
                    reverse=True
                )

                for req in queue:
                    rid = req['id']

                    # parse request
                    body = json.loads(req['body'])
                    model = body['params']['model']

                    # if whitelist enabled and model not in it continue
                    if (len(self.model_whitelist) > 0 and
                        not model in self.model_whitelist):
                        continue

                    # if blacklist contains model skip
                    if model in self.model_blacklist:
                        continue

                    my_results = [res['id'] for res in self._snap['my_results']]
                    if rid not in my_results:
                        statuses = self._snap['requests'][rid]

                        if len(statuses) == 0:
                            binary = await self.conn.get_input_data(req['binary_data'])

                            hash_str = (
                                str(req['nonce'])
                                +
                                req['body']
                                +
                                req['binary_data']
                            )
                            logging.info(f'hashing: {hash_str}')
                            request_hash = sha256(hash_str.encode('utf-8')).hexdigest()

                            # TODO: validate request

                            # perform work
                            logging.info(f'working on {body}')

                            resp = await self.conn.begin_work(rid)
                            if 'code' in resp:
                                logging.info(f'probably being worked on already... skip.')

                            else:
                                try:
                                    match self.backend:
                                        case 'sync-on-thread':
                                            self.mm._should_cancel = self.should_cancel_work
                                            img_sha, img_raw = await trio.to_thread.run_sync(
                                                partial(
                                                    self.mm.compute_one,
                                                    rid,
                                                    body['method'], body['params'], binary=binary
                                                )
                                            )

                                        case 'tractor':
                                            async def _should_cancel_oracle():
                                                while True:
                                                    await trio.sleep(1)
                                                    if (await self.should_cancel_work(rid)):
                                                        raise DGPUInferenceCancelled

                                            async with (
                                                trio.open_nursery() as trio_n,
                                                tractor.open_nursery() as tractor_n
                                            ):
                                                trio_n.start_soon(_should_cancel_oracle)
                                                portal = await tractor_n.run_in_actor(
                                                    _tractor_static_compute_one,
                                                    name='tractor-cuda-mp',
                                                    request_id=rid,
                                                    method=body['method'],
                                                    params=body['params'],
                                                    binary=binary
                                                )
                                                img_sha, img_raw = await portal.result()
                                                trio_n.cancel_scope.cancel()

                                        case _:
                                            raise DGPUComputeError(f'Unsupported backend {self.backend}')

                                    ipfs_hash = await self.conn.publish_on_ipfs(img_raw)

                                    await self.conn.submit_work(rid, request_hash, img_sha, ipfs_hash)

                                except BaseException as e:
                                    traceback.print_exc()
                                    await self.conn.cancel_work(rid, str(e))

                                finally:
                                    break

                    else:
                        logging.info(f'request {rid} already beign worked on, skip...')

                await trio.sleep(1)

        except KeyboardInterrupt:
            ...
