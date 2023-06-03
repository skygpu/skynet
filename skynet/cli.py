#!/usr/bin/python

import os
import time
import json
import logging
import random

from typing import Optional
from datetime import datetime, timedelta
from functools import partial

import trio
import asks
import click
import docker
import asyncio
import requests

from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, collect_stdout
from leap.hyperion import HyperionAPI

from .db import open_new_database
from .ipfs import open_ipfs_node
from .config import *
from .nodeos import open_cleos, open_nodeos
from .constants import ALGOS
from .frontend.telegram import run_skynet_telegram


@click.group()
def skynet(*args, **kwargs):
    pass


@click.command()
@click.option('--model', '-m', default='midj')
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=512)
@click.option('--height', '-h', default=512)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def txt2img(*args, **kwargs):
    from . import utils
    _, hf_token, _, cfg = init_env_from_config()
    utils.txt2img(hf_token, **kwargs)

@click.command()
@click.option('--model', '-m', default='midj')
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--input', '-i', default='input.png')
@click.option('--output', '-o', default='output.png')
@click.option('--strength', '-Z', default=1.0)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def img2img(model, prompt, input, output, strength, guidance, steps, seed):
    from . import utils
    _, hf_token, _, cfg = init_env_from_config()
    utils.img2img(
        hf_token,
        model=model,
        prompt=prompt,
        img_path=input,
        output=output,
        strength=strength,
        guidance=guidance,
        steps=steps,
        seed=seed
    )

@click.command()
@click.option('--input', '-i', default='input.png')
@click.option('--output', '-o', default='output.png')
@click.option('--model', '-m', default='weights/RealESRGAN_x4plus.pth')
def upscale(input, output, model):
    from . import utils
    utils.upscale(
        img_path=input,
        output=output,
        model_path=model)


@skynet.command()
def download():
    from . import utils
    _, hf_token, _, cfg = init_env_from_config()
    utils.download_all_models(hf_token)

@skynet.command()
@click.option(
    '--account', '-A', default=None)
@click.option(
    '--permission', '-P', default=None)
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--reward', '-r', default='20.0000 GPU')
@click.option('--algo', '-a', default='midj')
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=512)
@click.option('--height', '-h', default=512)
@click.option('--guidance', '-g', default=10)
@click.option('--step', '-s', default=26)
@click.option('--seed', '-S', default=None)
@click.option('--upscaler', '-U', default='x4')
def enqueue(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    reward: str,
    **kwargs
):
    key, account, permission = load_account_info(
        key, account, permission)
    with open_cleos(node_url, key=key) as cleos:
        if not kwargs['seed']:
            kwargs['seed'] = random.randint(0, 10e9)

        req = json.dumps({
            'method': 'diffuse',
            'params': kwargs
        })
        binary = ''

        ec, out = cleos.push_action(
            'telos.gpu', 'enqueue', [account, req, binary, reward], f'{account}@{permission}'
        )

        print(collect_stdout(out))
        assert ec == 0

@skynet.command()
@click.option('--loglevel', '-l', default='INFO', help='Logging level')
@click.option(
    '--account', '-A', default='telos.gpu')
@click.option(
    '--permission', '-P', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
def clean(
    loglevel: str,
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
):
    logging.basicConfig(level=loglevel)
    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    trio.run(
        partial(
            cleos.a_push_action,
            'telos.gpu',
            'clean',
            {},
            account, key, permission=permission
        )
    )

@skynet.command()
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
def queue(node_url: str):
    resp = requests.post(
        f'{node_url}/v1/chain/get_table_rows',
        json={
            'code': 'telos.gpu',
            'table': 'queue',
            'scope': 'telos.gpu',
            'json': True
        }
    )
    print(json.dumps(resp.json(), indent=4))

@skynet.command()
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.argument('request-id')
def status(node_url: str, request_id: int):
    resp = requests.post(
        f'{node_url}/v1/chain/get_table_rows',
        json={
            'code': 'telos.gpu',
            'table': 'status',
            'scope': request_id,
            'json': True
        }
    )
    print(json.dumps(resp.json(), indent=4))

@skynet.command()
@click.option(
    '--account', '-a', default='telegram1')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.argument('request-id')
def dequeue(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    request_id: int
):
    key, account, permission = load_account_info(
        key, account, permission)
    with open_cleos(node_url, key=key) as cleos:
        ec, out = cleos.push_action(
            'telos.gpu', 'dequeue', [account, request_id], f'{account}@{permission}'
        )

        print(collect_stdout(out))
        assert ec == 0

@skynet.command()
@click.option(
    '--account', '-a', default='telegram1')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--verifications', '-v', default=1)
@click.option(
    '--token-contract', '-c', default='eosio.token')
@click.option(
    '--token-symbol', '-S', default='4,GPU')
def config(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    verifications: int,
    token_contract: str,
    token_symbol: str
):
    key, account, permission = load_account_info(
        key, account, permission)
    with open_cleos(node_url, key=key) as cleos:
        ec, out = cleos.push_action(
            'telos.gpu', 'config', [verifications, token_contract, token_symbol], f'{account}@{permission}'
        )

        print(collect_stdout(out))
        assert ec == 0

@skynet.command()
@click.option(
    '--account', '-a', default='telegram1')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.argument('quantity')
def deposit(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    quantity: str
):
    key, account, permission = load_account_info(
        key, account, permission)
    with open_cleos(node_url, key=key) as cleos:
        ec, out = cleos.transfer_token(account, 'telos.gpu', quantity)

        print(collect_stdout(out))
        assert ec == 0

@skynet.group()
def run(*args, **kwargs):
    pass

@run.command()
def db():
    logging.basicConfig(level=logging.INFO)
    with open_new_database(cleanup=False) as db_params:
        container, passwd, host = db_params
        logging.info(('skynet', passwd, host))

@run.command()
def nodeos():
    logging.basicConfig(filename='skynet-nodeos.log', level=logging.INFO)
    with open_nodeos(cleanup=False):
        ...

@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--account', '-a', default='testworker1')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--auto-withdraw', '-w', default=True)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--ipfs-url', '-n', default='/ip4/169.197.140.154/udp/4001/quic/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv')
@click.option(
    '--algos', '-A', default=json.dumps(['midj', 'ink']))
def dgpu(
    loglevel: str,
    account: str,
    permission: str,
    key: str | None,
    auto_withdraw: bool,
    node_url: str,
    ipfs_url: str,
    algos: list[str]
):
    from .dgpu import open_dgpu_node

    key, account, permission = load_account_info(
        key, account, permission)

    trio.run(
        partial(
            open_dgpu_node,
            account, permission,
            CLEOS(None, None, url=node_url, remote=node_url),
            ipfs_url,
            auto_withdraw=auto_withdraw,
            key=key, initial_algos=json.loads(algos)
    ))


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--account', '-a', default='telegram')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--hyperion-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--ipfs-url', '-n', default='/ip4/169.197.140.154/udp/4001/quic/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv')
@click.option(
    '--db-host', '-h', default='localhost:5432')
@click.option(
    '--db-user', '-u', default='skynet')
@click.option(
    '--db-pass', '-u', default='password')
def telegram(
    loglevel: str,
    account: str,
    permission: str,
    key: str | None,
    hyperion_url: str,
    ipfs_url: str,
    node_url: str,
    db_host: str,
    db_user: str,
    db_pass: str
):
    logging.basicConfig(level=loglevel)

    key, account, permission = load_account_info(
        key, account, permission)

    _, _, tg_token, cfg = init_env_from_config()
    asyncio.run(
        run_skynet_telegram(
            tg_token,
            account,
            permission,
            node_url,
            hyperion_url,
            db_host, db_user, db_pass,
            remote_ipfs_node=ipfs_url,
            key=key
    ))


class IPFSHTTP:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def pin(self, cid: str):
        return requests.post(
            f'{self.endpoint}/api/v0/pin/add',
            params={'arg': cid}
        )

    async def a_pin(self, cid: str):
        return await asks.post(
            f'{self.endpoint}/api/v0/pin/add',
            params={'arg': cid}
        )


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option('--name', '-n', default='skynet-ipfs', help='container name')
def ipfs(loglevel, name):
    logging.basicConfig(level=loglevel)
    with open_ipfs_node(name=name):
        ...

@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--ipfs-rpc', '-i', default='http://127.0.0.1:5001')
@click.option(
    '--hyperion-url', '-n', default='http://127.0.0.1:42001')
def pinner(loglevel, ipfs_rpc, hyperion_url):
    logging.basicConfig(level=loglevel)
    ipfs_node = IPFSHTTP(ipfs_rpc)
    hyperion = HyperionAPI(hyperion_url)

    pinned = set()
    async def _async_main():

        async def capture_enqueues(after: datetime):
            enqueues = await hyperion.aget_actions(
                account='telos.gpu',
                filter='telos.gpu:enqueue',
                sort='desc',
                after=after.isoformat(),
                limit=1000
            )

            logging.info(f'got {len(enqueues["actions"])} enqueue actions.')

            cids = []
            for action in enqueues['actions']:
                cid = action['act']['data']['binary_data']
                if cid and cid not in pinned:
                    pinned.add(cid)
                    cids.append(cid)

            return cids

        async def capture_submits(after: datetime):
            submits = await hyperion.aget_actions(
                account='telos.gpu',
                filter='telos.gpu:submit',
                sort='desc',
                after=after.isoformat(),
                limit=1000
            )

            logging.info(f'got {len(submits["actions"])} submits actions.')

            cids = []
            for action in submits['actions']:
                cid = action['act']['data']['ipfs_hash']
                if cid and cid not in pinned:
                    pinned.add(cid)
                    cids.append(cid)

            return cids

        async def task_pin(cid: str):
            logging.info(f'pinning {cid}...')
            resp = await ipfs_node.a_pin(cid)
            if resp.status_code != 200:
                logging.error(f'error pinning {cid}:\n{resp.text}')

            else:
                logging.info(f'pinned {cid}')

        try:
            async with trio.open_nursery() as n:
                while True:
                    now = datetime.now()
                    prev_second = now - timedelta(seconds=10)

                    # filter for the ones not already pinned
                    cids = [
                        *(await capture_enqueues(prev_second)),
                        *(await capture_submits(prev_second))
                    ]

                    # pin and remember (in parallel)
                    for cid in cids:
                        n.start_soon(task_pin, cid)

                    await trio.sleep(1)

        except KeyboardInterrupt:
            ...

    trio.run(_async_main)
