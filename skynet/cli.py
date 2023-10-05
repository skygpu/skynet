#!/usr/bin/python

import json
import logging
import random

from functools import partial

import click

from leap.sugar import Name, asset_from_str

from .config import *
from .constants import *


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

    config = load_skynet_ini()
    hf_token = load_key(config, 'skynet.dgpu', 'hf_token')
    hf_home = load_key(config, 'skynet.dgpu', 'hf_home')
    set_hf_vars(hf_token, hf_home)
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
    config = load_skynet_ini()
    hf_token = load_key(config, 'skynet.dgpu', 'hf_token')
    hf_home = load_key(config, 'skynet.dgpu', 'hf_home')
    set_hf_vars(hf_token, hf_home)
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
    config = load_skynet_ini()
    hf_token = load_key(config, 'skynet.dgpu', 'hf_token')
    hf_home = load_key(config, 'skynet.dgpu', 'hf_home')
    set_hf_vars(hf_token, hf_home)
    utils.download_all_models(hf_token)

@skynet.command()
@click.option(
    '--reward', '-r', default='20.0000 GPU')
@click.option('--jobs', '-j', default=1)
@click.option('--model', '-m', default='stabilityai/stable-diffusion-xl-base-1.0')
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=1024)
@click.option('--height', '-h', default=1024)
@click.option('--guidance', '-g', default=10)
@click.option('--step', '-s', default=26)
@click.option('--seed', '-S', default=None)
@click.option('--upscaler', '-U', default='x4')
@click.option('--binary_data', '-b', default='')
def enqueue(
    reward: str,
    jobs: int,
    **kwargs
):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_ini()

    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)

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
def clean(
    loglevel: str,
):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_ini()
    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')

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
def queue():
    import requests
    config = load_skynet_ini()
    node_url = load_key(config, 'skynet.user', 'node_url')
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
@click.argument('request-id')
def status(request_id: int):
    import requests
    config = load_skynet_ini()
    node_url = load_key(config, 'skynet.user', 'node_url')
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
@click.argument('request-id')
def dequeue(request_id: int):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_ini()
    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    res = trio.run(
        partial(
            cleos.a_push_action,
            'telos.gpu',
            'dequeue',
            {
                'user': Name(account),
                'request_id': int(request_id),
            },
            account, key, permission=permission
        )
    )
    print(res)


@skynet.command()
@click.option(
    '--token-contract', '-c', default='eosio.token')
@click.option(
    '--token-symbol', '-S', default='4,GPU')
def config(
    token_contract: str,
    token_symbol: str
):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_ini()

    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    res = trio.run(
        partial(
            cleos.a_push_action,
            'telos.gpu',
            'config',
            {
                'token_contract': token_contract,
                'token_symbol': token_symbol,
            },
            account, key, permission=permission
        )
    )
    print(res)


@skynet.command()
@click.argument('quantity')
def deposit(quantity: str):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_ini()

    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')
    cleos = CLEOS(None, None, url=node_url, remote=node_url)

    res = trio.run(
        partial(
            cleos.a_push_action,
            'telos.gpu',
            'transfer',
            {
                'sender': Name(account),
                'recipient': Name('telos.gpu'),
                'amount': asset_from_str(quantity),
                'memo': f'{account} transferred {quantity} to telos.gpu'
            },
            account, key, permission=permission
        )
    )
    print(res)


@skynet.group()
def run(*args, **kwargs):
    pass

@run.command()
def db():
    from .db import open_new_database

    logging.basicConfig(level=logging.INFO)
    with open_new_database(cleanup=False) as db_params:
        container, passwd, host = db_params
        logging.info(('skynet', passwd, host))

@run.command()
def nodeos():
    from .nodeos import open_nodeos

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
    import trio
    from .dgpu import open_dgpu_node

    logging.basicConfig(level=loglevel)

    config = load_skynet_ini(file_path=config_path)
    hf_token = load_key(config, 'skynet.dgpu', 'hf_token')
    hf_home = load_key(config, 'skynet.dgpu', 'hf_home')
    set_hf_vars(hf_token, hf_home)

    assert 'skynet.dgpu' in config

    trio.run(open_dgpu_node, config['skynet.dgpu'])


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--db-host', '-h', default='localhost:5432')
@click.option(
    '--db-user', '-u', default='skynet')
@click.option(
    '--db-pass', '-u', default='password')
def telegram(
    loglevel: str,
    db_host: str,
    db_user: str,
    db_pass: str
):
    import asyncio
    from .frontend.telegram import SkynetTelegramFrontend

    logging.basicConfig(level=loglevel)

    config = load_skynet_ini()
    tg_token = load_key(config, 'skynet.telegram', 'tg_token')

    key = load_key(config, 'skynet.user', 'key')
    account = load_key(config, 'skynet.user', 'account')
    permission = load_key(config, 'skynet.user', 'permission')
    node_url = load_key(config, 'skynet.user', 'node_url')
    hyperion_url = load_key(config, 'skynet.telegram', 'hyperion_url')

    ipfs_gateway_url = load_key(config, 'skynet.telegram', 'ipfs_gateway_url')
    ipfs_url = load_key(config, 'skynet.telegram', 'ipfs_url')

    async def _async_main():
        frontend = SkynetTelegramFrontend(
            tg_token,
            account,
            permission,
            node_url,
            hyperion_url,
            db_host, db_user, db_pass,
            ipfs_url,
            remote_ipfs_node=ipfs_gateway_url,
            key=key
        )

        async with frontend.open():
            await frontend.bot.infinity_polling()


    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option(
    '--db-host', '-h', default='localhost:5432')
@click.option(
    '--db-user', '-u', default='skynet')
@click.option(
    '--db-pass', '-u', default='password')
def discord(
    loglevel: str,
    db_host: str,
    db_user: str,
    db_pass: str
):
    import asyncio
    from .frontend.discord import SkynetDiscordFrontend

    logging.basicConfig(level=loglevel)

    config = load_skynet_ini()
    dc_token = load_key(config, 'skynet.discord', 'dc_token')

    key = load_key(config, 'skynet.discord', 'key')
    account = load_key(config, 'skynet.discord', 'account')
    permission = load_key(config, 'skynet.discord', 'permission')
    node_url = load_key(config, 'skynet.discord', 'node_url')
    hyperion_url = load_key(config, 'skynet.discord', 'hyperion_url')

    ipfs_gateway_url = load_key(config, 'skynet.discord', 'ipfs_gateway_url')
    ipfs_url = load_key(config, 'skynet.discord', 'ipfs_url')

    async def _async_main():
        frontend = SkynetDiscordFrontend(
            # dc_token,
            account,
            permission,
            node_url,
            hyperion_url,
            db_host, db_user, db_pass,
            ipfs_url,
            remote_ipfs_node=ipfs_gateway_url,
            key=key
        )

        async with frontend.open():
            await frontend.bot.start(dc_token)

    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
@click.option('--name', '-n', default='skynet-ipfs', help='container name')
@click.option('--peer', '-p', default=(), help='connect to peer', multiple=True, type=str)
def ipfs(loglevel, name, peer):
    from skynet.ipfs.docker import open_ipfs_node

    logging.basicConfig(level=loglevel)
    with open_ipfs_node(name=name, peers=peer):
        ...

@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
def pinner(loglevel):
    import trio
    from leap.hyperion import HyperionAPI
    from .ipfs import AsyncIPFSHTTP
    from .ipfs.pinner import SkynetPinner

    config = load_skynet_ini()
    hyperion_url = load_key(config, 'skynet.pinner', 'hyperion_url')
    ipfs_url = load_key(config, 'skynet.pinner', 'ipfs_url')

    logging.basicConfig(level=loglevel)
    ipfs_node = AsyncIPFSHTTP(ipfs_url)
    hyperion = HyperionAPI(hyperion_url)

    pinner = SkynetPinner(hyperion, ipfs_node)

    trio.run(pinner.pin_forever)
