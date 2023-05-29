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
import click
import docker
import asyncio
import requests

from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, collect_stdout
from leap.hyperion import HyperionAPI

from .db import open_new_database
from .ipfs import IPFSDocker
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
@click.option(
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    '--node-url', '-n', default='http://skynet.ancap.tech')
@click.option(
    '--ipfs-url', '-n', default='/ip4/169.197.142.4/tcp/4001/p2p/12D3KooWKHKPFuqJPeqYgtUJtfZTHvEArRX2qvThYBrjuTuPg2Nx')
@click.option(
    '--algos', '-A', default=json.dumps(['midj']))
def dgpu(
    loglevel: str,
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    ipfs_url: str,
    algos: list[str]
):
    from .dgpu import open_dgpu_node

    key, account, permission = load_account_info(
        key, account, permission)

    vtestnet = None
    try:
        dclient = docker.from_env()
        vtestnet = get_container(
            dclient,
            default_nodeos_image(),
            force_unique=True,
            detach=True,
            network='host',
            remove=True)

        cleos = CLEOS(dclient, vtestnet, url=node_url, remote=node_url)

        trio.run(
            partial(
                open_dgpu_node,
                account, permission,
                cleos,
                ipfs_url,
                key=key, initial_algos=json.loads(algos)
        ))

    finally:
        if vtestnet:
            vtestnet.stop()


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--account', '-a', default='telegram')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--hyperion-url', '-n', default='http://test1.us.telos.net:42001')
@click.option(
    '--node-url', '-n', default='http://skynet.ancap.tech')
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
    node_url: str,
    hyperion_url: str,
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
            key=key
    ))


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--container', '-c', default='ipfs_host')
@click.option(
    '--hyperion-url', '-n', default='http://127.0.0.1:42001')
def pinner(loglevel, container, hyperion_url):
    logging.basicConfig(level=loglevel)
    dclient = docker.from_env()

    container = dclient.containers.get(container)
    ipfs_node = IPFSDocker(container)
    hyperion = HyperionAPI(hyperion_url)

    last_pinned: dict[str, datetime] = {}

    def cleanup_pinned(now: datetime):
        for cid in set(last_pinned.keys()):
            ts = last_pinned[cid]
            if now - ts > timedelta(minutes=1):
                del last_pinned[cid]

    try:
        while True:
            # get all submits in the last minute
            now = datetime.now()
            half_min_ago = now - timedelta(seconds=30)
            submits = hyperion.get_actions(
                account='telos.gpu',
                filter='telos.gpu:submit',
                sort='desc',
                after=half_min_ago.isoformat()
            )

            # filter for the ones not already pinned
            actions = [
                action
                for action in submits['actions']
                if action['act']['data']['ipfs_hash']
                not in last_pinned
            ]

            # pin and remember
            for action in actions:
                cid = action['act']['data']['ipfs_hash']
                last_pinned[cid] = now

                ipfs_node.pin(cid)

                logging.info(f'pinned {cid}')

            cleanup_pinned(now)

            time.sleep(1)

    except KeyboardInterrupt:
        ...
