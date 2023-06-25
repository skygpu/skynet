#!/usr/bin/python

from functools import partial
import io
import json
import time
import logging

import asks
from PIL import Image

from contextlib import ExitStack
from contextlib import asynccontextmanager as acm

from leap.cleos import CLEOS
from leap.sugar import Checksum256, Name, asset_from_str
from skynet.constants import DEFAULT_DOMAIN

from skynet.dgpu.errors import DGPUComputeError
from skynet.ipfs import get_ipfs_file
from skynet.ipfs.docker import open_ipfs_node


async def failable(fn: partial, ret_fail=None):
    try:
        return await fn()

    except (
        asks.errors.RequestTimeout,
        json.JSONDecodeError
    ):
        return ret_fail


class SkynetGPUConnector:

    def __init__(self, config: dict):
        self.account = Name(config['account'])
        self.permission = config['permission']
        self.key = config['key']
        self.node_url = config['node_url']
        self.hyperion_url = config['hyperion_url']
        self.ipfs_url = config['ipfs_url']

        self.cleos = CLEOS(
            None, None, self.node_url, remote=self.node_url)

        self._exit_stack = ExitStack()

    def connect(self):
        self.ipfs_node = self._exit_stack.enter_context(
            open_ipfs_node())

    def disconnect(self):
        self._exit_stack.close()

    @acm
    async def open(self):
        self.connect()
        yield self
        self.disconnect()


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

    def publish_on_ipfs(self, raw_img: bytes):
        logging.info('publish_on_ipfs')
        img = Image.open(io.BytesIO(raw_img))
        img.save(f'ipfs-docker-staging/image.png')

        # check for connections to peers, reconnect if none
        peers = self.ipfs_node.check_connect()
        if peers == "":
            self.ipfs_node.connect(
                '/ip4/169.197.140.154/tcp/4001/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv')

        ipfs_hash = self.ipfs_node.add('image.png')

        self.ipfs_node.pin(ipfs_hash)

        return ipfs_hash

    async def get_input_data(self, ipfs_hash: str) -> bytes:
        if ipfs_hash == '':
            return b''

        resp = await get_ipfs_file(f'https://ipfs.{DEFAULT_DOMAIN}/ipfs/{ipfs_hash}/image.png')
        if not resp:
            raise DGPUComputeError('Couldn\'t gather input data from ipfs')

        return resp.raw
