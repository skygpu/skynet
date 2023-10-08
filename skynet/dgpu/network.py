#!/usr/bin/python

import io
import json
import time
import logging

from pathlib import Path
from functools import partial

import asks
import trio
import anyio

from PIL import Image, UnidentifiedImageError

from leap.cleos import CLEOS
from leap.sugar import Checksum256, Name, asset_from_str
from skynet.constants import DEFAULT_DOMAIN

from skynet.dgpu.errors import DGPUComputeError
from skynet.ipfs import AsyncIPFSHTTP, get_ipfs_file


REQUEST_UPDATE_TIME = 3


async def failable(fn: partial, ret_fail=None):
    try:
        return await fn()

    except (
        OSError,
        json.JSONDecodeError,
        asks.errors.RequestTimeout,
        asks.errors.BadHttpResponse,
        anyio.BrokenResourceError
    ):
        return ret_fail


class SkynetGPUConnector:

    def __init__(self, config: dict):
        self.account = Name(config['account'])
        self.permission = config['permission']
        self.key = config['key']

        self.node_url = config['node_url']
        self.hyperion_url = config['hyperion_url']

        self.cleos = CLEOS(
            None, None, self.node_url, remote=self.node_url)

        self.ipfs_gateway_url = None
        if 'ipfs_gateway_url' in config:
            self.ipfs_gateway_url = config['ipfs_gateway_url']
        self.ipfs_url = config['ipfs_url']

        self.ipfs_client = AsyncIPFSHTTP(self.ipfs_url)

        self._wip_requests = {}

    # blockchain helpers

    async def get_work_requests_last_hour(self):
        logging.info('get_work_requests_last_hour')
        return await failable(
            partial(
                self.cleos.aget_table,
                'telos.gpu', 'telos.gpu', 'queue',
                index_position=2,
                key_type='i64',
                lower_bound=int(time.time()) - 3600
            ), ret_fail=[])

    async def get_status_by_request_id(self, request_id: int):
        logging.info('get_status_by_request_id')
        return await failable(
            partial(
                self.cleos.aget_table,
                'telos.gpu', request_id, 'status'), ret_fail=[])

    async def get_global_config(self):
        logging.info('get_global_config')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'telos.gpu', 'telos.gpu', 'config'))

        if rows:
            return rows[0]
        else:
            return None

    async def get_worker_balance(self):
        logging.info('get_worker_balance')
        rows = await failable(
            partial(
                self.cleos.aget_table,
                'telos.gpu', 'telos.gpu', 'users',
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


    async def get_full_queue_snapshot(self):
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
        logging.info('begin_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'telos.gpu',
                'workbegin',
                {
                    'worker': self.account,
                    'request_id': request_id,
                    'max_workers': 2
                },
                self.account, self.key,
                permission=self.permission
            )
        )

    async def cancel_work(self, request_id: int, reason: str):
        logging.info('cancel_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                'telos.gpu',
                'workcancel',
                {
                    'worker': self.account,
                    'request_id': request_id,
                    'reason': reason
                },
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
                    'telos.gpu',
                    'withdraw',
                    {
                        'user': self.account,
                        'quantity': asset_from_str(balance)
                    },
                    self.account, self.key,
                    permission=self.permission
                )
            )

    async def find_my_results(self):
        logging.info('find_my_results')
        return await failable(
            partial(
                self.cleos.aget_table,
                'telos.gpu', 'telos.gpu', 'results',
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
                'telos.gpu',
                'submit',
                {
                    'worker': self.account,
                    'request_id': request_id,
                    'request_hash': Checksum256(request_hash),
                    'result_hash': Checksum256(result_hash),
                    'ipfs_hash': ipfs_hash
                },
                self.account, self.key,
                permission=self.permission
            )
        )

    # IPFS helpers
    async def publish_on_ipfs(self, raw_img: bytes):
        Path('ipfs-staging').mkdir(exist_ok=True)
        logging.info('publish_on_ipfs')
        img = Image.open(io.BytesIO(raw_img))
        img.save('ipfs-staging/image.png')

        if self.ipfs_gateway_url:
            # check peer connections, reconnect to skynet gateway if not
            gateway_id = Path(self.ipfs_gateway_url).name
            peers = await self.ipfs_client.peers()
            if gateway_id not in [p['Peer'] for p in peers]:
                await self.ipfs_client.connect(self.ipfs_gateway_url)

        file_info = await self.ipfs_client.add(Path('ipfs-staging/image.png'))
        file_cid = file_info['Hash']

        await self.ipfs_client.pin(file_cid)

        return file_cid

    async def get_input_data(self, ipfs_hash: str) -> bytes:
        if ipfs_hash == '':
            return b''

        results = {}
        ipfs_link = f'https://ipfs.{DEFAULT_DOMAIN}/ipfs/{ipfs_hash}'
        ipfs_link_legacy = ipfs_link + '/image.png'

        async with trio.open_nursery() as n:
            async def get_and_set_results(link: str):
                res = await get_ipfs_file(link, timeout=1)
                logging.info(f'got response from {link}')
                if not res or res.status_code != 200:
                    logging.warning(f'couldn\'t get ipfs binary data at {link}!')

                else:
                    try:
                        with Image.open(io.BytesIO(res.raw)):
                            results[link] = res.raw
                            n.cancel_scope.cancel()

                    except UnidentifiedImageError:
                        logging.warning(f'couldn\'t get ipfs binary data at {link}!')

            n.start_soon(
                get_and_set_results, ipfs_link)
            n.start_soon(
                get_and_set_results, ipfs_link_legacy)

        png_img = None
        if ipfs_link_legacy in results:
            png_img = results[ipfs_link_legacy]

        if ipfs_link in results:
            png_img = results[ipfs_link]

        if not png_img:
            raise DGPUComputeError('Couldn\'t gather input data from ipfs')

        return png_img
