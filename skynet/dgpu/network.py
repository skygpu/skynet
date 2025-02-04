#!/usr/bin/python

import io
import json
import time
import logging

from pathlib import Path
from functools import partial

import trio
import leap
import anyio
import httpx

from PIL import (
    Image,
    # UnidentifiedImageError,  # TODO, remove?
)

from leap.cleos import CLEOS
from leap.protocol import Asset
from skynet.constants import (
    DEFAULT_IPFS_DOMAIN,
    GPU_CONTRACT_ABI,
)

from skynet.ipfs import (
    AsyncIPFSHTTP,
    get_ipfs_file,
)
# TODO, remove?
# from skynet.dgpu.errors import DGPUComputeError


REQUEST_UPDATE_TIME: int = 3


# TODO, consider using the `outcome` lib instead?
# - it's already purpose built for exactly this, boxing (async)
#  function invocations..
# |_ https://outcome.readthedocs.io/en/latest/api.html#outcome.capture
async def failable(
    fn: partial,
    ret_fail=None,
):
    try:
        return await fn()
    except (
        OSError,
        json.JSONDecodeError,
        anyio.BrokenResourceError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        leap.errors.TransactionPushError
    ):
        return ret_fail


# TODO, again the prefix XD
# -[ ] better name then `GPUConnector` ??
# |_ `Compute[Net]IO[Mngr]`
class SkynetGPUConnector:
    '''
    An API for connecting to and conducting various "high level"
    network-service operations in the skynet.

    - skynet user account creds
    - hyperion API
    - IPFs client
    - CLEOS client

    '''
    def __init__(self, config: dict):
        # TODO, why these extra instance vars for an (unsynced)
        # copy of the `config` state?
        self.account = config['account']
        self.permission = config['permission']
        self.key = config['key']

        # TODO, neither of these instance vars are used anywhere in
        # methods? so why are they set on this type?
        self.node_url = config['node_url']
        self.hyperion_url = config['hyperion_url']

        self.cleos = CLEOS(endpoint=self.node_url)
        self.cleos.load_abi('gpu.scd', GPU_CONTRACT_ABI)

        self.ipfs_gateway_url = None
        if 'ipfs_gateway_url' in config:
            self.ipfs_gateway_url = config['ipfs_gateway_url']
        self.ipfs_url = config['ipfs_url']

        self.ipfs_client = AsyncIPFSHTTP(self.ipfs_url)

        self.ipfs_domain = DEFAULT_IPFS_DOMAIN
        if 'ipfs_domain' in config:
            self.ipfs_domain = config['ipfs_domain']

        self._wip_requests = {}

    # blockchain helpers

    async def get_work_requests_last_hour(self):
        logging.info('get_work_requests_last_hour')
        return await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'queue',
                index_position=2,
                key_type='i64',
                lower_bound=int(time.time()) - 3600
            ), ret_fail=[])

    async def get_status_by_request_id(self, request_id: int):
        logging.info('get_status_by_request_id')
        return await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', request_id, 'status'), ret_fail=[])

    async def get_global_config(self):
        logging.info('get_global_config')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'config'))

        if rows:
            return rows[0]
        else:
            return None

    async def get_worker_balance(self):
        logging.info('get_worker_balance')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'users',
                index_position=1,
                key_type='name',
                lower_bound=self.account,
                upper_bound=self.account
            ))

        if rows:
            return rows[0]['balance']
        else:
            return None

    async def get_competitors_for_req(self, request_id: int) -> set:
        competitors = [
            status['worker']
            for status in
            (await self.get_status_by_request_id(request_id))
            if status['worker'] != self.account
        ]
        logging.info(f'competitors: {competitors}')
        return set(competitors)

    # TODO, considery making this a NON-method and instead
    # handing in the `snap['queue']` output beforehand?
    # -> since that call is the only usage of `self`?
    async def get_full_queue_snapshot(self):
        '''
        Keep in-sync with latest (telos chain's smart-contract) table
        state by polling (currently with period 1s).

        '''
        snap = {
            'requests': {},
            'my_results': []
        }

        snap['queue'] = await self.get_work_requests_last_hour()

        async def _run_and_save(d, key: str, fn, *args, **kwargs):
            d[key] = await fn(*args, **kwargs)

        async with trio.open_nursery() as n:
            n.start_soon(_run_and_save, snap, 'my_results', self.find_my_results)
            for req in snap['queue']:
                n.start_soon(
                    _run_and_save, snap['requests'], req['id'], self.get_status_by_request_id, req['id'])

        return snap

    async def begin_work(self, request_id: int):
        '''
        Publish to the bc that the worker is beginning a model-computation
        step.

        '''
        logging.info('begin_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'workbegin',
                list({
                    'worker': self.account,
                    'request_id': request_id,
                    'max_workers': 2
                }.values()),
                self.account, self.key,
                permission=self.permission
            )
        )

    async def cancel_work(self, request_id: int, reason: str):
        logging.info('cancel_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'workcancel',
                list({
                    'worker': self.account,
                    'request_id': request_id,
                    'reason': reason
                }.values()),
                self.account, self.key,
                permission=self.permission
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
                        'user': self.account,
                        'quantity': Asset.from_str(balance)
                    }.values()),
                    self.account, self.key,
                    permission=self.permission
                )
            )

    async def find_my_results(self):
        logging.info('find_my_results')
        return await failable(
            partial(
                self.cleos.aget_table,
                'gpu.scd', 'gpu.scd', 'results',
                index_position=4,
                key_type='name',
                lower_bound=self.account,
                upper_bound=self.account
            )
        )

    async def submit_work(
        self,
        request_id: int,
        request_hash: str,
        result_hash: str,
        ipfs_hash: str
    ):
        logging.info('submit_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'gpu.scd',
                'submit',
                list({
                    'worker': self.account,
                    'request_id': request_id,
                    'request_hash': request_hash,
                    'result_hash': result_hash,
                    'ipfs_hash': ipfs_hash
                }.values()),
                self.account, self.key,
                permission=self.permission
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

        if self.ipfs_gateway_url:
            # check peer connections, reconnect to skynet gateway if not
            gateway_id = Path(self.ipfs_gateway_url).name
            peers = await self.ipfs_client.peers()
            if gateway_id not in [p['Peer'] for p in peers]:
                await self.ipfs_client.connect(self.ipfs_gateway_url)

        file_info = await self.ipfs_client.add(Path(target_file))
        file_cid = file_info['Hash']

        await self.ipfs_client.pin(file_cid)

        return file_cid

    async def get_input_data(self, ipfs_hash: str) -> Image:
        '''
        Retrieve an input (image) from the IPFs layer.

        Normally used to retreive seed (visual) content previously
        generated/validated by the network to be fed to some
        consuming AI model.

        '''
        link = f'https://{self.ipfs_domain}/ipfs/{ipfs_hash}'

        res = await get_ipfs_file(link, timeout=1)
        logging.info(f'got response from {link}')
        if not res or res.status_code != 200:
            logging.warning(f'couldn\'t get ipfs binary data at {link}!')

        # attempt to decode as image
        input_data = Image.open(io.BytesIO(res.raw))

        return input_data
