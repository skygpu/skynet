import io
import json
import time
import logging
from pathlib import Path
from typing import AsyncGenerator
from functools import partial

import trio
import leap
import anyio
import httpx
import outcome
from PIL import Image
from leap.cleos import CLEOS
from leap.protocol import Asset
from skynet.dgpu.tui import maybe_update_tui
from skynet.config import DgpuConfig as Config
from skynet.constants import (
    DEFAULT_IPFS_DOMAIN,
    GPU_CONTRACT_ABI,
)

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

        self._tables = {
            'queue': [],
            'requests': {},
            'results': []
        }

        maybe_update_tui(lambda tui: tui.set_header_text(new_worker_name=self.config.account))


    # blockchain helpers

    async def get_work_requests_last_hour(self):
        logging.info('get_work_requests_last_hour')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'queue',
                index_position=2,
                key_type='i64',
                lower_bound=int(time.time()) - 3600
            ), ret_fail=[])

        logging.info(f'found {len(rows)} requests on queue')
        return rows

    async def get_status_by_request_id(self, request_id: int):
        logging.info('get_status_by_request_id')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', request_id, 'status'), ret_fail=[])

        logging.info(f'found status for workers: {[r["worker"] for r in rows]}')
        return rows

    async def get_global_config(self):
        logging.info('get_global_config')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'config'))

        if rows:
            cfg = rows[0]
            logging.info(f'config found: {cfg}')
            return cfg
        else:
            logging.error('global config not found, is the contract initialized?')
            return None

    async def get_worker_balance(self):
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
            b = rows[0]['balance']
            logging.info(f'balance: {b}')
            return b
        else:
            logging.info('no balance info found')
            return None

    async def get_full_queue_snapshot(self):
        '''
        Keep in-sync with latest (telos chain's smart-contract) table
        state by polling (currently with period 1s).

        '''
        snap = {
            'requests': {},
            'results': []
        }

        snap['queue'] = await self.get_work_requests_last_hour()

        async def _run_and_save(d, key: str, fn, *args, **kwargs):
            d[key] = await fn(*args, **kwargs)

        async with trio.open_nursery() as n:
            n.start_soon(_run_and_save, snap, 'results', self.find_results)
            for req in snap['queue']:
                n.start_soon(
                    _run_and_save, snap['requests'], req['id'], self.get_status_by_request_id, req['id'])


        maybe_update_tui(lambda tui: tui.network_update(snap))

        return snap

    async def iter_poll_update(self, poll_time: float) -> AsyncGenerator[dict, None]:
        '''
        Long running task, olls gpu contract tables yields latest table rows

        '''
        while True:
            start_time = time.time()
            self._tables = await self.get_full_queue_snapshot()
            elapsed = time.time() - start_time
            yield self._tables
            await trio.sleep(max(poll_time - elapsed, 0.1))

    async def should_cancel_work(self, request_id: int) -> bool:
        logging.info('should cancel work?')
        if request_id not in self._tables['requests']:
            logging.info(f'request #{request_id} no longer in queue, likely its been filled by another worker, cancelling work...')
            return True

        competitors = set([
            status['worker']
            for status in self._tables['requests'][request_id]
            if status['worker'] != self.config.account
        ])
        logging.info(f'competitors: {competitors}')
        should_cancel = bool(self.config.non_compete & competitors)
        logging.info(f'cancel: {should_cancel}')
        return should_cancel

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

    async def find_results(self):
        logging.info('find_results')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'results',
                index_position=4,
                key_type='name',
                lower_bound=self.config.account,
                upper_bound=self.config.account
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
