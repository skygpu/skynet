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
from leap.protocol import Asset
from skynet.constants import DEFAULT_IPFS_DOMAIN

from skynet.ipfs import AsyncIPFSHTTP, get_ipfs_file
from skynet.dgpu.errors import DGPUComputeError


REQUEST_UPDATE_TIME = 3

gpu_abi = {
    "version": "eosio::abi/1.2",
    "types": [],
    "structs": [
        {
            "name": "account",
            "base": "",
            "fields": [
                {"name": "user", "type": "name"},
                {"name": "balance", "type": "asset"},
                {"name": "nonce", "type": "uint64"}
            ]
        },
        {
            "name": "card",
            "base": "",
            "fields": [
                {"name": "id", "type": "uint64"},
                {"name": "owner", "type": "name"},
                {"name": "card_name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "total_memory", "type": "uint64"},
                {"name": "mp_count", "type": "uint32"},
                {"name": "extra", "type": "string"}
            ]
        },
        {
            "name": "clean",
            "base": "",
            "fields": []
        },
        {
            "name": "config",
            "base": "",
            "fields": [
                {"name": "token_contract", "type": "name"},
                {"name": "token_symbol", "type": "symbol"}
            ]
        },
        {
            "name": "dequeue",
            "base": "",
            "fields": [
                {"name": "user", "type": "name"},
                {"name": "request_id", "type": "uint64"}
            ]
        },
        {
            "name": "enqueue",
            "base": "",
            "fields": [
                {"name": "user", "type": "name"},
                {"name": "request_body", "type": "string"},
                {"name": "binary_data", "type": "string"},
                {"name": "reward", "type": "asset"},
                {"name": "min_verification", "type": "uint32"}
            ]
        },
        {
            "name": "gcfgstruct",
            "base": "",
            "fields": [
                {"name": "token_contract", "type": "name"},
                {"name": "token_symbol", "type": "symbol"}
            ]
        },
        {
            "name": "submit",
            "base": "",
            "fields": [
                {"name": "worker", "type": "name"},
                {"name": "request_id", "type": "uint64"},
                {"name": "request_hash", "type": "checksum256"},
                {"name": "result_hash", "type": "checksum256"},
                {"name": "ipfs_hash", "type": "string"}
            ]
        },
        {
            "name": "withdraw",
            "base": "",
            "fields": [
                {"name": "user", "type": "name"},
                {"name": "quantity", "type": "asset"}
            ]
        },
        {
            "name": "work_request_struct",
            "base": "",
            "fields": [
                {"name": "id", "type": "uint64"},
                {"name": "user", "type": "name"},
                {"name": "reward", "type": "asset"},
                {"name": "min_verification", "type": "uint32"},
                {"name": "nonce", "type": "uint64"},
                {"name": "body", "type": "string"},
                {"name": "binary_data", "type": "string"},
                {"name": "timestamp", "type": "time_point_sec"}
            ]
        },
        {
            "name": "work_result_struct",
            "base": "",
            "fields": [
                {"name": "id", "type": "uint64"},
                {"name": "request_id", "type": "uint64"},
                {"name": "user", "type": "name"},
                {"name": "worker", "type": "name"},
                {"name": "result_hash", "type": "checksum256"},
                {"name": "ipfs_hash", "type": "string"},
                {"name": "submited", "type": "time_point_sec"}
            ]
        },
        {
            "name": "workbegin",
            "base": "",
            "fields": [
                {"name": "worker", "type": "name"},
                {"name": "request_id", "type": "uint64"},
                {"name": "max_workers", "type": "uint32"}
            ]
        },
        {
            "name": "workcancel",
            "base": "",
            "fields": [
                {"name": "worker", "type": "name"},
                {"name": "request_id", "type": "uint64"},
                {"name": "reason", "type": "string"}
            ]
        },
        {
            "name": "worker",
            "base": "",
            "fields": [
                {"name": "account", "type": "name"},
                {"name": "joined", "type": "time_point_sec"},
                {"name": "left", "type": "time_point_sec"},
                {"name": "url", "type": "string"}
            ]
        },
        {
            "name": "worker_status_struct",
            "base": "",
            "fields": [
                {"name": "worker", "type": "name"},
                {"name": "status", "type": "string"},
                {"name": "started", "type": "time_point_sec"}
            ]
        }
    ],
    "actions": [
        {"name": "clean", "type": "clean", "ricardian_contract": ""},
        {"name": "config", "type": "config", "ricardian_contract": ""},
        {"name": "dequeue", "type": "dequeue", "ricardian_contract": ""},
        {"name": "enqueue", "type": "enqueue", "ricardian_contract": ""},
        {"name": "submit", "type": "submit", "ricardian_contract": ""},
        {"name": "withdraw", "type": "withdraw", "ricardian_contract": ""},
        {"name": "workbegin", "type": "workbegin", "ricardian_contract": ""},
        {"name": "workcancel", "type": "workcancel", "ricardian_contract": ""}
    ],
    "tables": [
        {
            "name": "cards",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "card"
        },
        {
            "name": "gcfgstruct",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "gcfgstruct"
        },
        {
            "name": "queue",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "work_request_struct"
        },
        {
            "name": "results",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "work_result_struct"
        },
        {
            "name": "status",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "worker_status_struct"
        },
        {
            "name": "users",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "account"
        },
        {
            "name": "workers",
            "index_type": "i64",
            "key_names": [],
            "key_types": [],
            "type": "worker"
        }
    ],
    "ricardian_clauses": [],
    "error_messages": [],
    "abi_extensions": [],
    "variants": [],
    "action_results": []
}



async def failable(fn: partial, ret_fail=None):
    try:
        return await fn()

    except (
        OSError,
        json.JSONDecodeError,
        asks.errors.RequestTimeout,
        asks.errors.BadHttpResponse,
        anyio.BrokenResourceError
    ) as e:
        return ret_fail


class SkynetGPUConnector:

    def __init__(self, config: dict):
        self.account = config['account']
        self.permission = config['permission']
        self.key = config['key']

        self.node_url = config['node_url']
        self.hyperion_url = config['hyperion_url']

        self.cleos = CLEOS(endpoint=self.node_url)
        self.cleos.load_abi('gpu.scd', gpu_abi)

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
        link = f'https://{self.ipfs_domain}/ipfs/{ipfs_hash}'

        res = await get_ipfs_file(link, timeout=1)
        logging.info(f'got response from {link}')
        if not res or res.status_code != 200:
            logging.warning(f'couldn\'t get ipfs binary data at {link}!')

        # attempt to decode as image
        input_data = Image.open(io.BytesIO(res.raw))

        return input_data
