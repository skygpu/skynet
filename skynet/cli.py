#!/usr/bin/python

import json
import logging
import random

from functools import partial

import trio
import asks
import click
import asyncio
import requests

from leap.cleos import CLEOS
from leap.sugar import collect_stdout, Name, asset_from_str
from leap.hyperion import HyperionAPI

from skynet.ipfs import IPFSHTTP


from .db import open_new_database
from .config import *
from .nodeos import open_cleos, open_nodeos
from .constants import *
from .frontend.telegram import SkynetTelegramFrontend
from .frontend.discord import SkynetDiscordFrontend


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
    _, hf_token, _, _ = init_env_from_config()
    utils.txt2img(hf_token, **kwargs)

@click.command()
@click.option('--model', '-m', default=list(MODELS.keys())[0])
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
    _, hf_token, _, _ = init_env_from_config()
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
    _, hf_token, _, _ = init_env_from_config()
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
@click.option('--jobs', '-j', default=1)
@click.option('--model', '-m', default='prompthero/openjourney')
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=512)
@click.option('--height', '-h', default=512)
@click.option('--guidance', '-g', default=10)
@click.option('--step', '-s', default=26)
@click.option('--seed', '-S', default=None)
@click.option('--upscaler', '-U', default='x4')
@click.option('--binary_data', '-b', default='')
def enqueue(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    reward: str,
    jobs: int,
    **kwargs
):
    key, account, permission = load_account_info(
        'user', key, account, permission)

    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)

    with open_cleos(node_url, key=key) as cleos:
        async def enqueue_n_jobs():
            for i in range(jobs):
                if not kwargs['seed']:
                    kwargs['seed'] = random.randint(0, 10e9)

                req = json.dumps({
                    'method': 'diffuse',
                    'params': kwargs
                })
                binary = kwargs['binary_data']

                res = await cleos.a_push_action(
                    'telos.gpu',
                    'enqueue',
                    {
                        'user': Name(account),
                        'request_body': req,
                        'binary_data': binary,
                        'reward': asset_from_str(reward),
                        'min_verification': 1
                    },
                    account, key, permission,
                )
                print(res)
        trio.run(enqueue_n_jobs)


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
    key, account, permission = load_account_info(
        'user', key, account, permission)

    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)

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
    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)
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
    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)
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
    '--account', '-a', default='telegram')
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
        'user', key, account, permission)

    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)

    with open_cleos(node_url, key=key) as cleos:
        res = trio.run(cleos.a_push_action,
            'telos.gpu',
            'dequeue',
            {
                'user': Name(account),
                'request_id': int(request_id),
            },
            account, key, permission,
        )
        print(res)


@skynet.command()
@click.option(
    '--account', '-a', default='telos.gpu')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='https://skynet.ancap.tech')
@click.option(
    '--token-contract', '-c', default='eosio.token')
@click.option(
    '--token-symbol', '-S', default='4,GPU')
def config(
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    token_contract: str,
    token_symbol: str
):
    key, account, permission = load_account_info(
        'user', key, account, permission)

    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)
    with open_cleos(node_url, key=key) as cleos:
        res = trio.run(cleos.a_push_action,
            'telos.gpu',
            'config',
            {
                'token_contract': token_contract,
                'token_symbol': token_symbol,
            },
            account, key, permission,
        )
        print(res)


@skynet.command()
@click.option(
    '--account', '-a', default='telegram')
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
        'user', key, account, permission)

    node_url, _, _ = load_endpoint_info(
        'user', node_url, None, None)

    with open_cleos(node_url, key=key) as cleos:
        res = trio.run(cleos.a_push_action,
            'eosio.token',
            'transfer',
            {
                'sender': Name(account),
                'recipient': Name('telos.gpu'),
                'amount': asset_from_str(quantity),
                'memo': f'{account} transferred {quantity} to telos.gpu'
            },
            account, key, permission,
        )
        print(res)


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
@click.option('--loglevel', '-l', default='INFO', help='Logging level')
@click.option(
    '--config-path', '-c', default='skynet.ini')
def dgpu(
    loglevel: str,
    config_path: str
):
    from .dgpu import open_dgpu_node

    logging.basicConfig(level=loglevel)

    config = load_skynet_ini(file_path=config_path)

    assert 'skynet.dgpu' in config

    trio.run(open_dgpu_node, config['skynet.dgpu'])


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--account', '-a', default='telegram')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--hyperion-url', '-y', default=f'https://{DEFAULT_DOMAIN}')
@click.option(
    '--node-url', '-n', default=f'https://{DEFAULT_DOMAIN}')
@click.option(
    '--ipfs-url', '-i', default=DEFAULT_IPFS_REMOTE)
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

    _, _, tg_token, _ = init_env_from_config()

    key, account, permission = load_account_info(
        'telegram', key, account, permission)

    node_url, _, ipfs_url = load_endpoint_info(
        'telegram', node_url, None, None)

    async def _async_main():
        frontend = SkynetTelegramFrontend(
            tg_token,
            account,
            permission,
            node_url,
            hyperion_url,
            db_host, db_user, db_pass,
            remote_ipfs_node=ipfs_url,
            key=key
        )

        async with frontend.open():
            await frontend.bot.infinity_polling()


    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--account', '-a', default=None)
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--hyperion-url', '-y', default=f'https://{DEFAULT_DOMAIN}')
@click.option(
    '--node-url', '-n', default=f'https://{DEFAULT_DOMAIN}')
@click.option(
    '--ipfs-url', '-i', default=DEFAULT_IPFS_REMOTE)
@click.option(
    '--db-host', '-h', default='localhost:5432')
@click.option(
    '--db-user', '-u', default='skynet')
@click.option(
    '--db-pass', '-u', default='password')
def discord(
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

    _, _, _, dc_token = init_env_from_config()

    key, account, permission = load_account_info(
        'discord', key, account, permission)

    node_url, _, ipfs_url = load_endpoint_info(
        'discord', node_url, None, None)

    async def _async_main():
        frontend = SkynetDiscordFrontend(
            # dc_token,
            account,
            permission,
            node_url,
            hyperion_url,
            # db_host, db_user, db_pass,
            remote_ipfs_node=ipfs_url,
            key=key
        )

        async with frontend.open():
            await frontend.bot.start(dc_token)

    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option('--name', '-n', default='skynet-ipfs', help='container name')
def ipfs(loglevel, name):
    from skynet.ipfs.docker import open_ipfs_node

    logging.basicConfig(level=loglevel)
    with open_ipfs_node(name=name):
        ...

@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--ipfs-rpc', '-i', default='http://127.0.0.1:5001')
@click.option(
    '--hyperion-url', '-y', default='http://127.0.0.1:42001')
def pinner(loglevel, ipfs_rpc, hyperion_url):
    from .ipfs.pinner import SkynetPinner

    logging.basicConfig(level=loglevel)
    ipfs_node = IPFSHTTP(ipfs_rpc)
    hyperion = HyperionAPI(hyperion_url)

    pinner = SkynetPinner(hyperion, ipfs_node)

    trio.run(pinner.pin_forever)
