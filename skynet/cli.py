#!/usr/bin/python

import os
import json

from typing import Optional
from functools import partial

import trio
import click

from .dgpu import open_dgpu_node
from .utils import txt2img
from .constants import ALGOS


@click.group()
def skynet(*args, **kwargs):
    pass

@skynet.command()
@click.option('--model', '-m', default=ALGOS['midj'])
@click.option(
    '--prompt', '-p', default='a red tractor in a wheat field')
@click.option('--output', '-o', default='output.png')
@click.option('--width', '-w', default=512)
@click.option('--height', '-h', default=512)
@click.option('--guidance', '-g', default=10.0)
@click.option('--steps', '-s', default=26)
@click.option('--seed', '-S', default=None)
def txt2img(*args
#     model: str,
#     prompt: str,
#     output: str
#     width: int, height: int,
#     guidance: float,
#     steps: int,
#     seed: Optional[int]
):
    assert 'HF_TOKEN' in os.environ
    txt2img(
        os.environ['HF_TOKEN'], *args)

@skynet.group()
def run(*args, **kwargs):
    pass

@run.command()
@click.option('--loglevel', '-l', default='warning', help='Logging level')
@click.option(
    '--key', '-k', default='dgpu')
@click.option(
    '--cert', '-c', default='whitelist/dgpu')
@click.option(
    '--algos', '-a', default=None)
def dgpu(
    loglevel: str,
    key: str,
    cert: str,
    algos: Optional[str]
):
    trio.run(
        partial(
            open_dgpu_node,
            cert,
            key_name=key,
            initial_algos=json.loads(algos)
    ))
