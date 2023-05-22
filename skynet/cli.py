#!/usr/bin/python
import importlib.util
torch_enabled = importlib.util.find_spec('torch') != None

import os
import json
import logging

from typing import Optional
from functools import partial

import trio
import click
import docker
import asyncio

from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container

if torch_enabled:
    from . import utils
    from .dgpu import open_dgpu_node

from .db import open_new_database
from .config import *
from .nodeos import open_nodeos
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


@skynet.command()
def download():
    _, hf_token, _, cfg = init_env_from_config()
    utils.download_all_models(hf_token)


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
    logging.basicConfig(level=logging.INFO)
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
    '--node-url', '-n', default='http://test1.us.telos.net:42000')
@click.option(
    '--algos', '-A', default=json.dumps(['midj']))
def dgpu(
    loglevel: str,
    account: str,
    permission: str,
    key: str | None,
    node_url: str,
    algos: list[str]
):
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
            cleos, key=key, initial_algos=json.loads(algos)
    ))

    vtestnet.stop()


@run.command()
@click.option('--loglevel', '-l', default='warning', help='logging level')
@click.option(
    '--account', '-a', default='telegram1')
@click.option(
    '--permission', '-p', default='active')
@click.option(
    '--key', '-k', default=None)
@click.option(
    '--node-url', '-n', default='http://test1.us.telos.net:42000')
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
    db_host: str,
    db_user: str,
    db_pass: str
):
    _, _, tg_token, cfg = init_env_from_config()
    asyncio.run(
        run_skynet_telegram(
            tg_token,
            account,
            permission,
            node_url,
            db_host, db_user, db_pass,
            key=key
    ))
