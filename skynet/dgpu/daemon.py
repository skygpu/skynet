#!/usr/bin/python

import json
import random
import logging
import time
import traceback

from hashlib import sha256
from datetime import datetime
from functools import partial

import trio

from quart import jsonify
from quart_trio import QuartTrio as Quart

from skynet.constants import MODELS, VERSION

from skynet.dgpu.errors import *
from skynet.dgpu.compute import SkynetMM
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

        self.account = config['account']

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

        self._benchmark = []
        self._last_benchmark = None
        self._last_generation_ts = None

    def _get_benchmark_speed(self) -> float:
        if not self._last_benchmark:
            return 0

        start = self._last_benchmark[0]
        end = self._last_benchmark[-1]

        elapsed = end - start
        its = len(self._last_benchmark)
        speed = its / elapsed

        logging.info(f'{elapsed} s total its: {its}, at {speed} it/s ')

        return speed

    async def should_cancel_work(self, request_id: int):
        self._benchmark.append(time.time())
        competitors = set([
            status['worker']
            for status in self._snap['requests'][request_id]
            if status['worker'] != self.account
        ])
        return bool(self.non_compete & competitors)


    async def snap_updater_task(self):
        while True:
            self._snap = await self.conn.get_full_queue_snapshot()
            await trio.sleep(1)

    async def generate_api(self):
        app = Quart(__name__)

        @app.route('/')
        async def health():
            return jsonify(
                account=self.account,
                version=VERSION,
                last_generation_ts=self._last_generation_ts,
                last_generation_speed=self._get_benchmark_speed()
            )

        return app

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

                    # if model not known
                    if model not in MODELS:
                        logging.warning(f'Unknown model {model}')
                        continue

                    # if whitelist enabled and model not in it continue
                    if (len(self.model_whitelist) > 0 and
                        not model in self.model_whitelist):
                        continue

                    # if blacklist contains model skip
                    if model in self.model_blacklist:
                        continue

                    my_results = [res['id'] for res in self._snap['my_results']]
                    if rid not in my_results and rid in self._snap['requests']:
                        statuses = self._snap['requests'][rid]

                        if len(statuses) == 0:
                            binary, input_type = await self.conn.get_input_data(req['binary_data'])

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
                                    output_type = 'png'
                                    if 'output_type' in body['params']:
                                        output_type = body['params']['output_type']

                                    output = None
                                    output_hash = None
                                    match self.backend:
                                        case 'sync-on-thread':
                                            self.mm._should_cancel = self.should_cancel_work
                                            output_hash, output = await trio.to_thread.run_sync(
                                                partial(
                                                    self.mm.compute_one,
                                                    rid,
                                                    body['method'], body['params'],
                                                    input_type=input_type,
                                                    binary=binary
                                                )
                                            )

                                        case _:
                                            raise DGPUComputeError(f'Unsupported backend {self.backend}')
                                    self._last_generation_ts = datetime.now().isoformat()
                                    self._last_benchmark = self._benchmark
                                    self._benchmark = []

                                    ipfs_hash = await self.conn.publish_on_ipfs(output, typ=output_type)

                                    await self.conn.submit_work(rid, request_hash, output_hash, ipfs_hash)

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
