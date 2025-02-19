import json
import logging

import click

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
    from skynet.dgpu import utils

    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.txt2img(config.dgpu.hf_token, **kwargs)


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
    from skynet.dgpu import utils
    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.img2img(
        config.dgpu.hf_token,
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
    from skynet.dgpu import utils
    config = load_skynet_toml()
    set_hf_vars(config.dgpu.hf_token, config.dgpu.hf_home)
    utils.inpaint(
        config.dgpu.hf_token,
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
    from skynet.dgpu import utils
    utils.upscale(
        img_path=input,
        output=output,
        model_path=model)


@skynet.command()
def download():
    from skynet.dgpu import utils
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
    from skynet.frontend.chatbot.db import open_new_database

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
def telegram(
    loglevel: str,
):
    import asyncio
    from skynet.frontend.chatbot.telegram import TelegramChatbot
    from skynet.frontend.chatbot.db import FrontendUserDB

    logging.basicConfig(level=loglevel)

    config = load_skynet_toml().telegram

    async def _async_main():
        async with FrontendUserDB(
            config.db_user,
            config.db_pass,
            config.db_host,
            config.db_name
        ) as db:
            bot = TelegramChatbot(config, db)
            await bot.run()

    asyncio.run(_async_main())


@run.command()
@click.option('--loglevel', '-l', default='INFO', help='logging level')
def discord(
    loglevel: str,
):
    import asyncio
    from skynet.frontend.chatbot.discord import DiscordChatbot
    from skynet.frontend.chatbot.db import FrontendUserDB

    logging.basicConfig(level=loglevel)

    config = load_skynet_toml().discord

    async def _async_main():
        async with FrontendUserDB(
            config.db_user,
            config.db_pass,
            config.db_host,
            config.db_name
        ) as db:
            bot = DiscordChatbot(config, db)
            await bot.run()

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
