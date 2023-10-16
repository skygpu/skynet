#!/usr/bin/python

import json
import logging
import random

from functools import partial

import click

from leap.sugar import Name, ListArgument, asset_from_str, symbol_from_str
import msgspec

from skynet.protocol import ComputeRequest, ParamsStruct, RequestRow

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

    config = load_skynet_toml()
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
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
    config = load_skynet_toml()
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
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
    config = load_skynet_toml()
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)
    utils.download_all_models(hf_token, hf_home)

@skynet.command()
@click.option(
    '--reward', '-r', default='20.0000 GPU')
@click.option('--jobs', '-j', default=1)
@click.option('--model', '-m', default='stabilityai/stable-diffusion-xl-base-1.0')
@click.option(
    '--prompt', '-p',
    default='cyberpunk skynet terminator skull a post impressionist oil painting with muted colors authored by Paul Cézanne, Paul Gauguin, Vincent van Gogh, Georges Seurat')
@click.option('--guidance', '-g', default=10)
@click.option('--step', '-s', default=26)
@click.option('--width', '-w', default=1024)
@click.option('--height', '-h', default=1024)
@click.option('--seed', '-S', default=None)
@click.option('--input', '-i', multiple=True)
@click.option('--strength', '-Z', default=None)
def enqueue(
    reward: str,
    jobs: int,
    model: str,
    prompt: str,
    guidance: float,
    step: int,
    **kwargs
):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_toml()
    logging.basicConfig(level='INFO')

    key = load_key(config, 'skynet.user.key')
    account = load_key(config, 'skynet.user.account')
    permission = load_key(config, 'skynet.user.permission')
    node_url = load_key(config, 'skynet.node_url')

    contract = load_key(config, 'skynet.contract')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)

    inputs = kwargs['input']
    if len(inputs) > 0:
        del kwargs['width']
        del kwargs['height']

    del kwargs['input']

    if not kwargs['strength']:
        if len(inputs) > 0:
            raise ValueError('strength -Z param required if input data passed')

        del kwargs['strength']

    else:
        kwargs['strength'] = float(kwargs['strength'])

    async def enqueue_n_jobs():
        actions = []
        for _ in range(jobs):
            if kwargs['seed']:
                seed = kwargs['seed']
            else:
                seed = random.randint(0, 10e9)

            _kwargs = kwargs.copy()
            _kwargs['generator'] = seed
            del _kwargs['seed']

            request = ComputeRequest(
                method='diffuse',
                params=ParamsStruct(
                    model=ModelParams(
                        name=model,
                        pipe_fqn='diffusers.DiffusionPipeline',
                        setup={'variant': 'fp16'}
                    ),
                    runtime_args=[prompt],
                    runtime_kwargs={
                        'guidance_scale': guidance,
                        'num_inference_steps': step,
                        **_kwargs
                    }
                )
            )

            req = msgspec.json.encode(request)

            actions.append({
                'account': contract,
                'name': 'enqueue',
                'data': [
                    Name(account),
                    ListArgument(req, 'uint8'),
                    ListArgument(inputs, 'string'),
                    asset_from_str(reward),
                    1
                ],
                'authorization': [{
                    'actor': account,
                    'permission': permission
                }]
            })

        # breakpoint()
        res = await cleos.a_push_actions(actions, key)
        print(res)

    trio.run(enqueue_n_jobs)


@skynet.command()
@click.option('--loglevel', '-l', default='INFO', help='Logging level')
def clean(
    loglevel: str,
):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_toml()
    key = load_key(config, 'skynet.user.key')
    account = load_key(config, 'skynet.user.account')
    permission = load_key(config, 'skynet.user.permission')
    node_url = load_key(config, 'skynet.node_url')
    contract = load_key(config, 'skynet.contract')

    logging.basicConfig(level=loglevel)
    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    trio.run(
        partial(
            cleos.a_push_action,
            contract,
            'clean',
            {},
            account, key, permission=permission
        )
    )

@skynet.command()
def queue():
    import requests
    config = load_skynet_toml()
    node_url = load_key(config, 'skynet.node_url')
    contract = load_key(config, 'skynet.contract')
    resp = requests.post(
        f'{node_url}/v1/chain/get_table_rows',
        json={
            'code': contract,
            'table': 'queue',
            'scope': contract,
            'json': True
        }
    ).json()

    # process hex body
    results = []
    for row in resp['rows']:
        req = row.copy()
        req['body'] = json.loads(bytes.fromhex(req['body']).decode())
        results.append(req)

    print(json.dumps(results, indent=4))


@skynet.command()
@click.argument('request-id')
def dequeue(request_id: int):
    import trio
    from leap.cleos import CLEOS

    config = load_skynet_toml()
    key = load_key(config, 'skynet.user.key')
    account = load_key(config, 'skynet.user.account')
    permission = load_key(config, 'skynet.user.permission')
    node_url = load_key(config, 'skynet.node_url')
    contract = load_key(config, 'skynet.contract')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    res = trio.run(
        partial(
            cleos.a_push_action,
            contract,
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
@click.argument(
    'token-contract', required=True)
@click.argument(
    'token-symbol', required=True)
@click.argument(
    'nonce', required=True)
def config(
    token_contract: str,
    token_symbol: str,
    nonce: int
):
    import trio
    from leap.cleos import CLEOS

    logging.basicConfig(level='INFO')
    config = load_skynet_toml()

    key = load_key(config, 'skynet.user.key')
    account = load_key(config, 'skynet.user.account')
    permission = load_key(config, 'skynet.user.permission')
    node_url = load_key(config, 'skynet.node_url')
    contract = load_key(config, 'skynet.contract')

    cleos = CLEOS(None, None, url=node_url, remote=node_url)
    res = trio.run(
        partial(
            cleos.a_push_action,
            contract,
            'config',
            {
                'token_contract': Name(token_contract),
                'token_symbol': symbol_from_str(token_symbol),
                'nonce': int(nonce)
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

    config = load_skynet_toml()

    key = load_key(config, 'skynet.user.key')
    account = load_key(config, 'skynet.user.account')
    permission = load_key(config, 'skynet.user.permission')
    node_url = load_key(config, 'skynet.node_url')
    contract = load_key(config, 'skynet.contract')
    cleos = CLEOS(None, None, url=node_url, remote=node_url)

    res = trio.run(
        partial(
            cleos.a_push_action,
            'eosio.token',
            'transfer',
            {
                'sender': Name(account),
                'recipient': Name(contract),
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
    '--config-path', '-c', default=DEFAULT_CONFIG_PATH)
def dgpu(
    loglevel: str,
    config_path: str
):
    import trio
    from .dgpu import open_dgpu_node

    logging.basicConfig(level=loglevel)

    config = load_skynet_toml(file_path=config_path)
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)

    assert 'skynet' in config
    assert 'dgpu' in config['skynet']

    trio.run(open_dgpu_node, config['skynet'])


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

    config = load_skynet_toml()
    tg_token = load_key(config, 'skynet.telegram.token')

    key = load_key(config, 'skynet.telegram.key')
    account = load_key(config, 'skynet.telegram.account')
    permission = load_key(config, 'skynet.telegram.permission')
    node_url = load_key(config, 'skynet.node_url')
    hyperion_url = load_key(config, 'skynet.hyperion_url')

    try:
        ipfs_gateway_url = load_key(config, 'skynet.ipfs_gateway_url')

    except ConfigParsingError:
        ipfs_gateway_url = None

    ipfs_url = load_key(config, 'skynet.ipfs_url')

    try:
        explorer_domain = load_key(config, 'skynet.explorer_domain')

    except ConfigParsingError:
        explorer_domain = DEFAULT_EXPLORER_DOMAIN

    try:
        ipfs_domain = load_key(config, 'skynet.ipfs_domain')

    except ConfigParsingError:
        ipfs_domain = DEFAULT_IPFS_DOMAIN

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
            key=key,
            explorer_domain=explorer_domain,
            ipfs_domain=ipfs_domain
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

    config = load_skynet_toml()
    dc_token = load_key(config, 'skynet.discord.token')

    key = load_key(config, 'skynet.discord.key')
    account = load_key(config, 'skynet.discord.account')
    permission = load_key(config, 'skynet.discord.permission')
    node_url = load_key(config, 'skynet.node_url')
    hyperion_url = load_key(config, 'skynet.hyperion_url')

    ipfs_gateway_url = load_key(config, 'skynet.ipfs_gateway_url')
    ipfs_url = load_key(config, 'skynet.ipfs_url')

    try:
        explorer_domain = load_key(config, 'skynet.explorer_domain')

    except ConfigParsingError:
        explorer_domain = DEFAULT_EXPLORER_DOMAIN

    try:
        ipfs_domain = load_key(config, 'skynet.ipfs_domain')

    except ConfigParsingError:
        ipfs_domain = DEFAULT_IPFS_DOMAIN

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
            key=key,
            explorer_domain=explorer_domain,
            ipfs_domain=ipfs_domain
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

    config = load_skynet_toml()
    hyperion_url = load_key(config, 'skynet.hyperion_url')
    ipfs_url = load_key(config, 'skynet.ipfs_url')

    logging.basicConfig(level=loglevel)
    ipfs_node = AsyncIPFSHTTP(ipfs_url)
    hyperion = HyperionAPI(hyperion_url)

    pinner = SkynetPinner(hyperion, ipfs_node)

    trio.run(pinner.pin_forever)
