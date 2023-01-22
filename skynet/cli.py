#!/usr/bin/python
import importlib.util
torch_enabled = importlib.util.find_spec('torch') != None

import os
import json

from typing import Optional
from functools import partial

import trio
import click
import trio_asyncio

if torch_enabled:
    from . import utils
    from .dgpu import open_dgpu_node

from .brain import run_skynet
from .config import *
from .constants import ALGOS, DEFAULT_RPC_ADDR, DEFAULT_DGPU_ADDR
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
    utils.upscale(
        img_path=input,
        output=output,
        model_path=model)


@skynet.group()
def run(*args, **kwargs):
    pass


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--host', '-H', default=DEFAULT_RPC_ADDR)
def brain(
    loglevel: str,
    host: str
):
    async def _run_skynet():
        async with run_skynet(
            rpc_address=host
        ):
            await trio.sleep_forever()

    trio.run(_run_skynet)


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--uid', '-u', required=True)
@click.option(
    '--key', '-k', default='dgpu.key')
@click.option(
    '--cert', '-c', default='whitelist/dgpu.cert')
@click.option(
    '--algos', '-a', default=json.dumps(['midj']))
@click.option(
    '--rpc', '-r', default=DEFAULT_RPC_ADDR)
@click.option(
    '--dgpu', '-d', default=DEFAULT_DGPU_ADDR)
def dgpu(
    loglevel: str,
    uid: str,
    key: str,
    cert: str,
    algos: str,
    rpc: str,
    dgpu: str
):
    trio.run(
        partial(
            open_dgpu_node,
            cert,
            uid,
            key_name=key,
            rpc_address=rpc,
            dgpu_address=dgpu,
            initial_algos=json.loads(algos)
    ))


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--key', '-k', default='telegram-frontend')
@click.option(
    '--cert', '-c', default='whitelist/telegram-frontend')
@click.option(
    '--rpc', '-r', default=DEFAULT_RPC_ADDR)
def telegram(
    loglevel: str,
    key: str,
    cert: str,
    rpc: str
):
    _, _, tg_token, cfg = init_env_from_config()
    trio_asyncio.run(
        partial(
            run_skynet_telegram,
            tg_token,
            key_name=key,
            cert_name=cert,
            rpc_address=rpc
    ))
