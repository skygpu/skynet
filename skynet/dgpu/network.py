#!/usr/bin/python

import io
import json
import time
import logging

from pathlib import Path
from functools import partial
from typing import Any, Coroutine

import asks
import trio
import anyio

from PIL import Image, UnidentifiedImageError

from leap.cleos import CLEOS
from leap.sugar import Checksum256, Name, asset_from_str
from skynet.constants import DEFAULT_IPFS_DOMAIN

from skynet.ipfs import AsyncIPFSHTTP, get_ipfs_file
from skynet.dgpu.errors import DGPUComputeError


REQUEST_UPDATE_TIME = 3


async def failable(
    fn: partial[Coroutine[Any, Any, Any]],
    ret_fail: Any | None = None
) -> Any:
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
        self.contract = config['contract']
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

        self.ipfs_domain = DEFAULT_IPFS_DOMAIN
        if 'ipfs_domain' in config:
            self.ipfs_domain = config['ipfs_domain']

        self.worker_url = ''
        if 'worker_url' in config:
            self.worker_url = config['worker_url']

        self._update_delta = 1
        self._cache: dict[str, tuple[float, Any]] = {}

    async def _cache_set(
        self,
        fn: partial[Coroutine[Any, Any, Any]],
        key: str
    ) -> Any:
        now = time.time()
        val = await fn()
        self._cache[key] = (now, val)

        return val

    def _cache_get(self, key: str, default: Any = None) -> Any:
        if key in self._cache:
            return self._cache[key][1]

        else:
            return default

    async def data_updater_task(self):
        tasks = (
            (self._get_work_requests_last_hour, 'queue'),
            (self._find_my_results, 'my_results'),
            (self._get_workers, 'workers')
        )

        while True:
            async with trio.open_nursery() as n:
                for task in tasks:
                    n.start_soon(self._cache_set, *task)

            await trio.sleep(self._update_delta)

    def get_queue(self):
        return self._cache_get('queue', default=[])

    def get_my_results(self):
        return self._cache_get('my_results', default=[])

    def get_workers(self):
        return self._cache_get('workers', default=[])

    def get_status_for_request(self, request_id: int) -> list[dict] | None:
        request: dict | None = next((
            req
            for req in self.get_queue()
            if req['id'] == request_id), None)

        if request:
            return request['status']

        else:
            return None

    def get_competitors_for_request(self, request_id: int) -> list[str] | None:
        status = self.get_status_for_request(request_id)
        if not status:
            return None

        return [
            s['worker']
            for s in status
            if s['worker'] != self.account
        ]

    async def _get_work_requests_last_hour(self) -> list[dict]:
        logging.info('get_work_requests_last_hour')
        return await failable(
            partial(
                self.cleos.aget_table,
                self.contract, self.contract, 'queue',
                index_position=2,
                order='asc',
                limit=1000
            ), ret_fail=[])

    async def _find_my_results(self):
        logging.info('find_my_results')
        return await failable(
            partial(
                self.cleos.aget_table,
                self.contract, self.contract, 'results',
                index_position=4,
                key_type='name',
                lower_bound=self.account,
                upper_bound=self.account
            )
        )

    async def _get_workers(self) -> list[dict]:
        logging.info('get_workers')
        return await failable(
            partial(
                self.cleos.aget_table,
                self.contract, self.contract, 'workers'
            )
        )

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
                self.contract, self.contract, 'users',
                index_position=1,
                key_type='name',
                lower_bound=self.account,
                upper_bound=self.account
            ))

        if rows:
            return rows[0]['balance']
        else:
            return None

    def get_on_chain_worker_info(self, worker: str):
        return next((
            w for w in self.get_workers()
            if w['account'] == w
        ), None)

    async def register_worker(self):
        logging.info(f'registering worker')
        return await failable(
            partial(
                self.cleos.a_push_action,
                self.contract,
                'regworker',
                {
                    'account': self.account,
                    'url': self.worker_url
                }
            )
        )

    async def add_card(
        self,
        card_name: str,
        version: str,
        total_memory: int,
        mp_count: int,
        extra: str,
        is_online: bool
    ):
        logging.info(f'adding card: {card_name} {version}')
        return await failable(
            partial(
                self.cleos.a_push_action,
                self.contract,
                'addcard',
                {
                    'worker': self.account,
                    'card_name': card_name,
                    'version': version,
                    'total_memory': total_memory,
                    'mp_count': mp_count,
                    'extra': extra,
                    'is_online': is_online
                }
            )
        )

    async def toggle_card(self, index: int):
        logging.info(f'toggle card {index}')
        return await failable(
            partial(
                self.cleos.a_push_action,
                self.contract,
                'togglecard',
                {'worker': self.account, 'index': index}
            )
        )

    async def flush_cards(self):
        logging.info('flushing cards...')
        return await failable(
            partial(
                self.cleos.a_push_action,
                self.contract,
                'flushcards',
                {'worker': self.account}
            )
        )

    async def begin_work(self, request_id: int):
        logging.info('begin_work')
        return await failable(
            partial(
                self.cleos.a_push_action,
                self.contract,
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
                self.contract,
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
                    self.contract,
                    'withdraw',
                    {
                        'user': self.account,
                        'quantity': asset_from_str(balance)
                    },
                    self.account, self.key,
                    permission=self.permission
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
                self.contract,
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

    async def get_input_data(self, ipfs_hash: str) -> tuple[bytes, str]:
        results = {}
        ipfs_link = f'https://{self.ipfs_domain}/ipfs/{ipfs_hash}'
        ipfs_link_legacy = ipfs_link + '/image.png'

        input_type = 'unknown'
        async with trio.open_nursery() as n:
            async def get_and_set_results(link: str):
                res = await get_ipfs_file(link, timeout=1)
                logging.info(f'got response from {link}')
                if not res or res.status_code != 200:
                    logging.warning(f'couldn\'t get ipfs binary data at {link}!')

                else:
                    try:
                        # attempt to decode as image
                        results[link] = Image.open(io.BytesIO(res.raw))
                        input_type = 'png'
                        n.cancel_scope.cancel()

                    except UnidentifiedImageError:
                        logging.warning(f'couldn\'t get ipfs binary data at {link}!')

            n.start_soon(
                get_and_set_results, ipfs_link)
            n.start_soon(
                get_and_set_results, ipfs_link_legacy)

        input_data = None
        if ipfs_link_legacy in results:
            input_data = results[ipfs_link_legacy]

        if ipfs_link in results:
            input_data = results[ipfs_link]

        if input_data == None:
            raise DGPUComputeError('Couldn\'t gather input data from ipfs')

        return input_data, input_type

    async def get_inputs(self, links: list[str]) -> list[tuple[bytes, str]]:
        results = {}
        async def _get_input(link: str) -> None:
            results[link] = await self.get_input_data(link)

        async with trio.open_nursery() as n:
            for link in links:
                n.start_soon(_get_input, link)

        return [results[link] for link in links]
