#!/usr/bin/python

import json
import time
import random
import logging
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
from skynet.protocol import ComputeRequest, ModelParams, ParamsStruct, RequestRow


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
        self.max_concurrent = (
            config['max_concurrent']
            if 'max_concurrent' in config else 0
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
        status = self.conn.get_status_for_request(request_id)
        competitors = [
            s.worker
            for s in status
            if s.worker != self.account
        ]
        return bool(self.non_compete & set(competitors))

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

    def find_best_requests(self) -> list[tuple[RequestRow, ComputeRequest]]:
        queue = self.conn.get_queue()

        for _ in range(3):
            random.shuffle(queue)

        queue = sorted(
            queue,
            key=lambda req: convert_reward_to_int(req.reward),
            reverse=True
        )

        requests = []
        for req in queue:
            rid = req.nonce

            # parse request
            try:
                req_json = json.loads(req.body)
                compute_request = ComputeRequest(**req_json)
                compute_request.params = ParamsStruct(**req_json['params'])
                compute_request.params.model = ModelParams(**req_json['params']['model'])
                model = compute_request.params.model.name

            except TypeError as e:
                logging.warning(f'Couldn\'t parse request: {e}')
                continue

            except json.JSONDecodeError as e:
                logging.warning(f'Couldn\'t parse request: {e}')
                continue

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

            my_results = [res.id for res in self.conn.get_my_results()]

            # if this worker already on it
            if rid in my_results:
                continue

            status = self.conn.get_status_for_request(rid)
            if status == None:
                continue

            if self.non_compete & set([
                s.worker
                for s in status
                if s.worker != self.account
            ]):
                continue

            if len(status) > self.max_concurrent:
                continue

            requests.append((req, compute_request))

        return requests

    async def sync_worker_on_chain_data(self, is_online: bool) -> bool:
        # check worker is registered
        me = self.conn.get_on_chain_worker_info(self.account)
        if not me:
            res = await self.conn.register_worker()
            if 'error' in res:
                raise DGPUDaemonError(f'Couldn\'t register worker! {out}')

            me = self.conn.get_on_chain_worker_info(self.account)
            if not me:
                raise DGPUDaemonError('Unknown error while registering')

        # find if reported on chain gpus match local
        found_difference = False
        for i in range(self.mm.num_gpus):
            chain_gpu = me.cards[i]

            gpu = self.mm.gpus[i]
            gpu_v = f'{gpu.major}.{gpu.minor}'

            found_difference = gpu.name != chain_gpu.card_name
            found_difference = gpu_v != chain_gpu.version
            found_difference = gpu.total_memory != chain_gpu.total_memory
            found_difference = gpu.multi_processor_count != chain_gpu.mp_count
            if found_difference:
                break

        # difference found, flush and re-report
        if found_difference:
            await self.conn.flush_cards()
            for i, gpu in enumerate(self.mm.gpus):
                res = await self.conn.add_card(
                    gpu.name, f'{gpu.major}.{gpu.minor}',
                    gpu.total_memory, gpu.multi_processor_count,
                    '',
                    is_online
                )
                if 'error' in res:
                    raise DGPUDaemonError(f'error while reporting card {i}')

        return found_difference

    async def all_gpu_set_online_flag(self, is_online: bool):
        me = self.conn.get_on_chain_worker_info(self.account)
        if not me:
            raise DGPUDaemonError('Couldn\'t find worker info!')

        for i, chain_gpu in enumerate(me.cards):
            if chain_gpu.is_online != is_online:
                await self.conn.toggle_card(i)

    async def serve_forever(self):

        diff = await self.sync_worker_on_chain_data(True)
        if not diff:
            await self.all_gpu_set_online_flag(True)

        try:
            while True:
                if self.auto_withdraw:
                    await self.conn.maybe_withdraw_all()

                requests = self.find_best_requests()

                if len(requests) > 0:
                    request, compute_request = requests[0]
                    rid = request.nonce
                    body = json.loads(request.body)
                    logging.info(f'trying to process req: {rid}')

                    hash_buf = (
                        str(request.nonce).encode()
                        +
                        request.body.encode()
                        +
                        b''.join([_in.encode() for _in in request.inputs])
                    )
                    logging.info(f'hashing str of length {len(hash_buf)}')
                    request_hash = sha256(hash_buf).hexdigest()

                    inputs = []
                    if len(request.inputs) > 0:
                        inputs = await self.conn.get_inputs(request.inputs)

                    # perform work
                    logging.info(f'working on {body}')

                    resp = await self.conn.begin_work(rid)
                    if 'code' in resp:
                        logging.info(f'probably being worked on already... skip.')

                    else:
                        try:
                            output_type = 'png'
                            if 'output_type' in compute_request.params.runtime_kwargs:
                                output_type = compute_request.params.runtime_kwargs['output_type']

                            outputs = []
                            match self.backend:
                                case 'sync-on-thread':
                                    self.mm._should_cancel = self.should_cancel_work
                                    outputs = await trio.to_thread.run_sync(
                                        partial(
                                            self.mm.compute_one,
                                            rid, compute_request,
                                            inputs=inputs
                                        )
                                    )

                                case _:
                                    raise DGPUComputeError(f'Unsupported backend {self.backend}')

                            self._last_generation_ts = datetime.now().isoformat()
                            self._last_benchmark = self._benchmark
                            self._benchmark = []

                            outputs = await self.conn.publish_on_ipfs(outputs)

                            await self.conn.submit_work(rid, request_hash, outputs)

                        except BaseException as e:
                            traceback.print_exc()
                            await self.conn.cancel_work(rid, str(e))

                await trio.sleep(1)

        except KeyboardInterrupt:
            ...

        await self.sync_worker_on_chain_data(False)
        await self.all_gpu_set_online_flag(False)
