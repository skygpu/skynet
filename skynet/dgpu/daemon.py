import json
import logging
import random
import time
from datetime import datetime
from functools import partial
from hashlib import sha256

import trio
from quart import jsonify
from quart_trio import QuartTrio as Quart

from skynet.constants import (
    MODELS,
    VERSION,
)
from skynet.dgpu.errors import (
    DGPUComputeError,
)
from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.network import SkynetGPUConnector


def convert_reward_to_int(reward_str):
    int_part, decimal_part = (
        reward_str.split('.')[0],
        reward_str.split('.')[1].split(' ')[0]
    )
    return int(int_part + decimal_part)


# prolly don't need the `Skynet` prefix since that's kinda implied ;p
class SkynetDGPUDaemon:
    '''
    The root "GPU daemon".

    Contains/manages underlying susystems:
    - a GPU connecto

    '''
    def __init__(
        self,
        mm: SkynetMM,
        conn: SkynetGPUConnector,
        config: dict
    ):
        self.mm: SkynetMM = mm
        self.conn: SkynetGPUConnector = conn
        self.auto_withdraw = (
            config['auto_withdraw']
            if 'auto_withdraw' in config else False
        )

        self.account: str = config['account']

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
            # ^and here i thot they were **my** results..
            # :sadcat:
        }

        self._benchmark: list[float] = []
        self._last_benchmark: list[float]|None = None
        self._last_generation_ts: str|None = None

    def _get_benchmark_speed(self) -> float:
        '''
        Return the (arithmetic) average work-iterations-per-second
        fconducted by this compute worker.

        '''
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
        '''
        Busy loop update the local `._snap: dict` table from

        '''
        while True:
            self._snap = await self.conn.get_full_queue_snapshot()
            await trio.sleep(1)

    # TODO, design suggestion, just make this a lazily accessed
    # `@class_property` if we're 3.12+
    # |_ https://docs.python.org/3/library/functools.html#functools.cached_property
    async def generate_api(self) -> Quart:
        '''
        Gen a `Quart`-compat web API spec which (for now) simply
        serves a small monitoring ep that reports,

        - iso-time-stamp of the last served model-output
        - the worker's average "compute-iterations-per-second"

        '''
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

    # TODO? this func is kinda big and maybe is better at module
    # level to reduce indentation?
    # -[ ] just pass `daemon: SkynetDGPUDaemon` vs. `self`
    async def maybe_serve_one(
        self,
        req: dict,
    ):
        rid = req['id']

        # parse request
        body = json.loads(req['body'])
        model = body['params']['model']

        # if model not known, ignore.
        if (
            model != 'RealESRGAN_x4plus'
            and
            model not in MODELS
        ):
            logging.warning(f'Unknown model {model}')
            return False

        # only handle whitelisted models
        if (
            len(self.model_whitelist) > 0
            and
            model not in self.model_whitelist
        ):
            return False

        # if blacklist contains model skip
        if model in self.model_blacklist:
            return False

        my_results = [res['id'] for res in self._snap['my_results']]
        if (
            rid not in my_results
            and
            rid in self._snap['requests']
        ):
            statuses = self._snap['requests'][rid]
            if len(statuses) == 0:
                inputs = []
                for _input in req['binary_data'].split(','):
                    if _input:
                        for _ in range(3):
                            try:
                                # user `GPUConnector` to IO with
                                # storage layer to seed the compute
                                # task.
                                img = await self.conn.get_input_data(_input)
                                inputs.append(img)
                                break

                            except BaseException:
                                logging.exception(
                                    'Model input error !?!\n'
                                )

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
                if not resp or 'code' in resp:
                    logging.info('probably being worked on already... skip.')

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
                                        inputs=inputs
                                    )
                                )

                            case _:
                                raise DGPUComputeError(
                                    f'Unsupported backend {self.backend}'
                                )

                        self._last_generation_ts: str = datetime.now().isoformat()
                        self._last_benchmark: list[float] = self._benchmark
                        self._benchmark: list[float] = []

                        ipfs_hash = await self.conn.publish_on_ipfs(output, typ=output_type)

                        await self.conn.submit_work(rid, request_hash, output_hash, ipfs_hash)

                    except BaseException as err:
                        logging.exception('Failed to serve model request !?\n')
                        # traceback.print_exc()  # TODO? <- replaced by above ya?
                        await self.conn.cancel_work(rid, str(err))

                    finally:
                        return True

        # TODO, i would inverse this case logic to avoid an indent
        # level in above block ;)
        else:
            logging.info(f'request {rid} already beign worked on, skip...')

    # TODO, as per above on `.maybe_serve_one()`, it's likely a bit
    # more *trionic* to define this all as a module level task-func
    # which operates on a `daemon: SkynetDGPUDaemon`?
    #
    # -[ ] keeps tasks-as-funcs style prominent
    # -[ ] avoids so much indentation due to methods
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
                    # TODO, as mentioned above just inline this once
                    # converted to a mod level func.
                    if (await self.maybe_serve_one(req)):
                        break

                await trio.sleep(1)

        except KeyboardInterrupt:
            ...
