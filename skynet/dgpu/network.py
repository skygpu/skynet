import io
import json
import time
import random
import logging
from pathlib import Path
from contextlib import asynccontextmanager as acm
from functools import partial

import trio
import leap
import anyio
import httpx
import outcome
import msgspec
from PIL import Image
from leap.cleos import CLEOS
from leap.protocol import Asset
from skynet.dgpu.tui import maybe_update_tui
from skynet.config import DgpuConfig as Config, load_skynet_toml
from skynet.types import (
    ConfigV0,
    BodyV0,
    RequestV0,
    WorkerStatusV0,
    ResultV0
)
from skynet.constants import GPU_CONTRACT_ABI

from skynet.ipfs import (
    AsyncIPFSHTTP,
    get_ipfs_file,
)


REQUEST_UPDATE_TIME: int = 3


async def failable(fn: partial, ret_fail=None):
    o = await outcome.acapture(fn)
    match o:
        case outcome.Error(error=(
            OSError() |
            json.JSONDecodeError() |
            anyio.BrokenResourceError() |
            httpx.ConnectError() |
            httpx.ConnectTimeout() |
            httpx.ReadError() |
            httpx.ReadTimeout() |
            leap.errors.TransactionPushError()
        )):
            return ret_fail

        case _:
            return o.unwrap()


class NetConnector:
    '''
    An API for connecting to and conducting various "high level"
    network-service operations in the skynet.

    - skynet user account creds
    - hyperion API
    - IPFs client
    - CLEOS client

    '''
    def __init__(self, config: Config):
        self.config = config
        self.cleos = CLEOS(endpoint=config.node_url)
        self.cleos.load_abi('gpu.scd', GPU_CONTRACT_ABI)

        self.ipfs_client = AsyncIPFSHTTP(config.ipfs_url)

        maybe_update_tui(lambda tui: tui.set_header_text(new_worker_name=self.config.account))


    # blockchain helpers

    async def get_work_requests_last_hour(self) -> list[RequestV0]:
        logging.info('get_work_requests_last_hour')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'queue',
                index_position=2,
                key_type='i64',
                lower_bound=int(time.time()) - 3600,
                resp_cls=RequestV0
            ), ret_fail=[])

        logging.info(f'found {len(rows)} requests on queue')
        return rows

    async def get_status_by_request_id(self, request_id: int) -> list[WorkerStatusV0]:
        logging.info('get_status_by_request_id')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', request_id, 'status', resp_cls=WorkerStatusV0), ret_fail=[])

        logging.info(f'found status for workers: {[r.worker for r in rows]}')
        return rows

    async def get_global_config(self) -> ConfigV0:
        logging.info('get_global_config')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'config',
                resp_cls=ConfigV0))

        if rows:
            cfg = rows[0]
            logging.info(f'config found: {cfg}')
            return cfg
        else:
            logging.error('global config not found, is the contract initialized?')
            return None

    async def get_worker_balance(self) -> str:
        logging.info('get_worker_balance')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'users',
                index_position=1,
                key_type='name',
                lower_bound=self.config.account,
                upper_bound=self.config.account
            ))

        if rows:
            b = rows[0].balance
            logging.info(f'balance: {b}')
            return b
        else:
            logging.info('no balance info found')
            return None

    async def begin_work(self, request_id: int):
        '''
        Publish to the bc that the worker is beginning a model-computation
        step.

        '''
        logging.info(f'begin_work on #{request_id}')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'workbegin',
                list({
                    'worker': self.config.account,
                    'request_id': request_id,
                    'max_workers': 2
                }.values()),
                self.config.account, self.config.key,
                permission=self.config.permission
            )
        )

    async def cancel_work(self, request_id: int, reason: str):
        logging.info(f'cancel_work on #{request_id}')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'workcancel',
                list({
                    'worker': self.config.account,
                    'request_id': request_id,
                    'reason': reason
                }.values()),
                self.config.account, self.config.key,
                permission=self.config.permission
            )
        )

    async def maybe_withdraw_all(self):
        logging.info('maybe_withdraw_all')
        balance = await self.get_worker_balance()
        if not balance:
            return

        balance_amount = float(balance.split(' ')[0])
        if balance_amount > 0:
            await failable(
                partial(
                    self.cleos.a_push_action,
                    'gpu.scd',
                    'withdraw',
                    list({
                        'user': self.config.account,
                        'quantity': Asset.from_str(balance)
                    }.values()),
                    self.config.account, self.config.key,
                    permission=self.config.permission
                )
            )

    async def find_results(self) -> list[ResultV0]:
        logging.info('find_results')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'results',
                index_position=4,
                key_type='name',
                lower_bound=self.config.account,
                upper_bound=self.config.account,
                resp_cls=ResultV0
            )
        )
        return rows

    async def submit_work(
        self,
        request_id: int,
        request_hash: str,
        result_hash: str,
        ipfs_hash: str
    ):
        logging.info(f'submit_work #{request_id}')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'submit',
                list({
                    'worker': self.config.account,
                    'request_id': request_id,
                    'request_hash': request_hash,
                    'result_hash': result_hash,
                    'ipfs_hash': ipfs_hash
                }.values()),
                self.config.account, self.config.key,
                permission=self.config.permission
            )
        )

    # IPFS helpers
    async def publish_on_ipfs(self, raw, typ: str = 'png'):
        Path('ipfs-staging').mkdir(exist_ok=True)
        logging.info('publish_on_ipfs')

        target_file = ''
        match typ:
            case 'png':
                raw: Image
                target_file = 'ipfs-staging/image.png'
                raw.save(target_file)

            case _:
                raise ValueError(f'Unsupported output type: {typ}')

        file_info = await self.ipfs_client.add(Path(target_file))
        file_cid = file_info['Hash']
        logging.info(f'added file to ipfs, CID: {file_cid}')

        await self.ipfs_client.pin(file_cid)
        logging.info(f'pinned {file_cid}')

        return file_cid

    async def get_input_data(self, ipfs_hash: str) -> Image:
        '''
        Retrieve an input (image) from the IPFs layer.

        Normally used to retreive seed (visual) content previously
        generated/validated by the network to be fed to some
        consuming AI model.

        '''
        link = f'https://{self.config.ipfs_domain}/ipfs/{ipfs_hash}'

        res = await get_ipfs_file(link, timeout=1)
        if not res or res.status_code != 200:
            logging.warning(f'couldn\'t get ipfs binary data at {link}!')

        # attempt to decode as image
        input_data = Image.open(io.BytesIO(res.read()))
        logging.info('decoded as image successfully')

        return input_data



def convert_reward_to_int(reward_str):
    int_part, decimal_part = (
        reward_str.split('.')[0],
        reward_str.split('.')[1].split(' ')[0]
    )
    return int(int_part + decimal_part)


class ContractState:

    def __init__(self, conn: NetConnector):
        self._conn = conn

        self._poll_index = 0

        self._queue: list[RequestV0] = []
        self._status_by_rid: dict[int, list[WorkerStatusV0]] = {}
        self._results: list[ResultV0] = []

        self._new_data = trio.Event()

    @property
    def poll_index(self) -> int:
        return self._poll_index

    async def _fetch_results(self):
        self._results = await self._conn.find_results()

    async def _fetch_statuses_for_id(self, rid: int):
        self._status_by_rid[rid] = await self._conn.get_status_by_request_id(rid)

    async def update_state(self):
        '''
        Get a "snapshot" of current contract table state

        '''
        # raw queue from chain
        _queue = await self._conn.get_work_requests_last_hour()

        # filter out invalids
        self._queue = []
        for req in _queue:
            try:
                msgspec.json.decode(req.body, type=BodyV0)
                self._queue.append(req)

            except msgspec.ValidationError:
                ...

        random.shuffle(self._queue)
        self._queue = sorted(
            self._queue,
            key=lambda req: convert_reward_to_int(req.reward),
            reverse=True
        )

        async with trio.open_nursery() as n:
            n.start_soon(self._fetch_results)
            for req in self._queue:
                n.start_soon(
                    self._fetch_statuses_for_id, req.id)


        maybe_update_tui(lambda tui: tui.network_update(self))

    async def wait_data_update(self):
        await self._new_data.wait()

    async def _state_update_task(self, poll_time: float):
        '''
        Long running task, polls gpu contract tables latest table rows,
        awakes any self._data_event waiters

        '''
        while True:
            start_time = time.time()
            await self.update_state()
            elapsed = time.time() - start_time
            self._new_data.set()
            await trio.sleep(max(poll_time - elapsed, 0.1))
            self._new_data = trio.Event()
            self._poll_index += 1

    # views into data

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    @property
    def first(self) -> RequestV0 | None:
        if len(self._queue) > 0:
            return self._queue[0]

        else:
            return None

    def competitors_for_id(self, request_id: int) -> set[str]:
        return set((
            status.worker
            for status in self._status_by_rid[request_id]
            if status.worker != self._conn.config.account
        ))

    # predicates

    def is_request_filled(self, request_id: int) -> bool:
        return request_id in [
            result.request_id for result in self._results
        ]

    def is_request_in_progress(self, request_id: int) -> bool:
        return request_id in self._status_by_rid

    def should_compete_for_id(self, request_id: int) -> bool:
        return bool(
            self._conn.config.non_compete &
            self.competitors_for_id(request_id)
        )

    async def should_cancel_work(self, request_id: int) -> bool:
        logging.info('should cancel work?')
        if request_id not in self._status_by_rid:
            logging.info(f'request #{request_id} no longer in queue, likely its been filled by another worker, cancelling work...')
            return True

        should_cancel = self.should_compete_for_id(request_id)
        logging.info(f'cancel: {should_cancel}')
        return should_cancel



__state_mngr = None

@acm
async def maybe_open_contract_state_mngr(conn: NetConnector):
    global __state_mngr

    if __state_mngr:
        yield __state_mngr
        return

    config = load_skynet_toml().dgpu

    mngr = ContractState(conn)
    async with trio.open_nursery() as n:
        await mngr.update_state()
        n.start_soon(mngr._state_update_task, config.poll_time)
        __state_mngr = mngr
        yield mngr
        n.cancel_scope.cancel()
