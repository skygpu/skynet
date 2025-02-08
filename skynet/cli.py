import json
import logging
import random

from functools import partial

import click

from leap.protocol import (
    Name,
    Asset,
)

from .config import (
    load_skynet_toml,
    set_hf_vars,
    ConfigParsingError,
)
from .constants import (
    # TODO, more conventional to make these private i'm pretty
    # sure according to pep8?
    DEFAULT_IPFS_DOMAIN,
    DEFAULT_EXPLORER_DOMAIN,
    DEFAULT_CONFIG_PATH,
    MODELS,
)


@click.group()
def skynet(*args, **kwargs):
    pass


@click.command()
@click.option('--model', '-m', default=list(MODELS.keys())[-1])
@click.option(
    '--prompt',
    '-p',
    default='a red old tractor in a sunny wheat field',
)
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=512)
@click.option('--height', '-h', default=512)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def txt2img(*args, **kwargs):
    from . import utils  # TODO? why here, import cycle?

    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.txt2img(hf_token, **kwargs)


@click.command()
@click.option(
    '--model',
    '-m',
    default=list(MODELS.keys())[-2]
)
@click.option(
    '--prompt',
    '-p',
    default='a red old tractor in a sunny wheat field',
)
@click.option('--input', '-i', default='input.png')
@click.option('--output', '-o', default='output.png')
@click.option('--strength', '-Z', default=1.0)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def img2img(model, prompt, input, output, strength, guidance, steps, seed):
    from . import utils
    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
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
@click.option('--model', '-m', default=list(MODELS.keys())[-3])
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--input', '-i', default='input.png')
@click.option('--mask', '-M', default='mask.png')
@click.option('--output', '-o', default='output.png')
@click.option('--strength', '-Z', default=1.0)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def inpaint(model, prompt, input, mask, output, strength, guidance, steps, seed):
    from . import utils
    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.inpaint(
        hf_token,
        model=model,
        prompt=prompt,
        img_path=input,
        mask_path=mask,
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
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.download_all_models(config.dgpu.hf_token, config.dgpu.hf_home)


@skynet.command()
def queue():
    import requests
    config = load_skynet_toml()
    node_url = config.user.node_url
    resp = requests.post(
        f'{node_url}/v1/chain/get_table_rows',
        json={
            'code': 'gpu.scd',
            'table': 'queue',
            'scope': 'gpu.scd',
            'json': True
        }
    )
    print(json.dumps(resp.json(), indent=4))

@skynet.command()
@click.argument('request-id')
def status(request_id: int):
    import requests
    config = load_skynet_toml()
    node_url = config.user.node_url
    resp = requests.post(
        f'{node_url}/v1/chain/get_table_rows',
        json={
            'code': 'gpu.scd',
            'table': 'status',
            'scope': request_id,
            'json': True
        }
    )
    print(json.dumps(resp.json(), indent=4))

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
@click.option('--loglevel', '-l', default='INFO', help='Logging level')
@click.option(
    '--config-path',
    '-c',
    default=DEFAULT_CONFIG_PATH,
)
def dgpu(
    loglevel: str,
    config_path: str
):
    import trio
    from .dgpu import _dgpu_main

    logging.basicConfig(level=loglevel)

    config = load_skynet_toml(file_path=config_path).dgpu
    set_hf_vars(config.hf_token, config.hf_home)

    trio.run(_dgpu_main, config)


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
    tg_token = config.telegram.tg_token

    key = config.telegram.key
    account = config.telegram.account
    permission = config.telegram.permission
    node_url = config.telegram.node_url
    hyperion_url = config.telegram.hyperion_url

    ipfs_url = config.telegram.ipfs_url

    try:
        explorer_domain = config.telegram.explorer_domain

    except ConfigParsingError:
        explorer_domain = DEFAULT_EXPLORER_DOMAIN

    try:
        ipfs_domain = config.telegram.ipfs_domain

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
    dc_token = config.discord.dc_token

    key = config.discord.key
    account = config.discord.account
    permission = config.discord.permission
    node_url = config.discord.node_url
    hyperion_url = config.discord.hyperion_url

    ipfs_url = config.discord.ipfs_url

    try:
        explorer_domain = config.discord.explorer_domain

    except ConfigParsingError:
        explorer_domain = DEFAULT_EXPLORER_DOMAIN

    try:
        ipfs_domain = config.discord.ipfs_domain

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
            key=key,
            explorer_domain=explorer_domain,
            ipfs_domain=ipfs_domain
        )

        async with frontend.open():
            await frontend.bot.start(dc_token)

    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
def pinner(loglevel):
    import trio
    from leap.hyperion import HyperionAPI
    from .ipfs import AsyncIPFSHTTP
    from .ipfs.pinner import SkynetPinner

    config = load_skynet_toml()
    hyperion_url = config.pinner.hyperion_url
    ipfs_url = config.pinner.ipfs_url

    logging.basicConfig(level=loglevel)
    ipfs_node = AsyncIPFSHTTP(ipfs_url)
    hyperion = HyperionAPI(hyperion_url)

    pinner = SkynetPinner(hyperion, ipfs_node)

    trio.run(pinner.pin_forever)
