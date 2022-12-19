#!/usr/bin/python

import os
import json

from typing import Optional
from functools import partial

import trio
import click

from . import utils
from .dgpu import open_dgpu_node
from .brain import run_skynet

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
@click.option(
    '--prompt', '-p', default='a red old tractor in a sunny wheat field')
@click.option('--input', '-i', default='input.png')
@click.option('--output', '-o', default='output.png')
@click.option('--steps', '-s', default=26)
def upscale(prompt, input, output, steps):
    assert 'HF_TOKEN' in os.environ
    utils.upscale(
        os.environ['HF_TOKEN'],
        prompt=prompt,
        img_path=input,
        output=output,
        steps=steps)


@skynet.group()
def run(*args, **kwargs):
    pass


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--host', '-h', default='localhost:5432')
@click.option(
    '--pass', '-p', default='password')
def skynet(
    loglevel: str,
    host: str,
    passw: str
):
    async def _run_skynet():
        async with run_skynet(
            db_host=host,
            db_pass=passw
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
    '--algos', '-a', default=None)
def dgpu(
    loglevel: str,
    uid: str,
    key: str,
    cert: str,
    algos: Optional[str]
):
    trio.run(
        partial(
            open_dgpu_node,
            cert,
            uid,
            key_name=key,
            initial_algos=json.loads(algos)
    ))


@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--key', '-k', default='telegram-frontend')
@click.option(
    '--cert', '-c', default='whitelist/telegram-frontend')
def telegram(
    loglevel: str,
    key: str,
    cert: str
):
    assert 'TG_TOKEN' in os.environ
    trio_asyncio.run(
        partial(
            run_skynet_telegram,
            os.environ['TG_TOKEN'],
            key_name=key,
            cert_name=cert
    ))
