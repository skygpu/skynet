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
    assert 'HF_TOKEN' in os.environ
    utils.txt2img(os.environ['HF_TOKEN'], **kwargs)

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
@click.option(
    '--host-dgpu', '-D', default=DEFAULT_DGPU_ADDR)
@click.option(
    '--db-host', '-h', default='localhost:5432')
@click.option(
    '--db-pass', '-p', default='password')
def brain(
    loglevel: str,
    host: str,
    host_dgpu: str,
    db_host: str,
    db_pass: str
):
    async def _run_skynet():
        async with run_skynet(
            db_host=db_host,
            db_pass=db_pass,
            rpc_address=host,
            dgpu_address=host_dgpu
        ):
            await trio.sleep_forever()

    trio_asyncio.run(_run_skynet)


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--uid', '-u', required=True)
@click.option(
    '--key', '-k', default='dgpu')
@click.option(
    '--cert', '-c', default='whitelist/dgpu')
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
    assert 'TG_TOKEN' in os.environ
    trio_asyncio.run(
        partial(
            run_skynet_telegram,
            os.environ['TG_TOKEN'],
            key_name=key,
            cert_name=cert,
            rpc_address=rpc
    ))
