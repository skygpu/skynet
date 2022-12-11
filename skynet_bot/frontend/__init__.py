#!/usr/bin/python

import json

from typing import Union
from contextlib import asynccontextmanager as acm

import pynng

from ..types import SkynetRPCRequest, SkynetRPCResponse
from ..constants import *


class ConfigRequestFormatError(BaseException):
    ...

class ConfigUnknownAttribute(BaseException):
    ...

class ConfigUnknownAlgorithm(BaseException):
    ...

class ConfigUnknownUpscaler(BaseException):
    ...

class ConfigSizeDivisionByEight(BaseException):
    ...


async def rpc_call(
    sock,
    uid: Union[int, str],
    method: str,
    params: dict = {}
):
    req = SkynetRPCRequest(
        uid=uid,
        method=method,
        params=params
    )
    await sock.asend(
        json.dumps(
            req.to_dict()).encode())

    return SkynetRPCResponse(
        **json.loads(
            (await sock.arecv_msg()).bytes.decode()))


@acm
async def open_skynet_rpc(rpc_address: str = DEFAULT_RPC_ADDR):
    with pynng.Req0(dial=rpc_address) as sock:
        async def _rpc_call(*args, **kwargs):
            return await rpc_call(sock, *args, **kwargs)

        yield _rpc_call


def validate_user_config_request(req: str):
    params = req.split(' ')

    if len(params) < 3:
        raise ConfigRequestFormatError('config request format incorrect')

    else:
        try:
            attr = params[1]

            if attr == 'algo':
                val = params[2]
                if val not in ALGOS:
                    raise ConfigUnknownAlgorithm(f'no algo named {val}')

            elif attr == 'step':
                val = int(params[2])
                val = max(min(val, MAX_STEP), MIN_STEP)

            elif attr  == 'width':
                val = max(min(int(params[2]), MAX_WIDTH), 16)
                if val % 8 != 0:
                    raise ConfigSizeDivisionByEight(
                        'size must be divisible by 8!')

            elif attr  == 'height':
                val = max(min(int(params[2]), MAX_HEIGHT), 16)
                if val % 8 != 0:
                    raise ConfigSizeDivisionByEight(
                        'size must be divisible by 8!')

            elif attr == 'seed':
                val = params[2]
                if val == 'auto':
                    val = None
                else:
                    val = int(params[2])

            elif attr == 'guidance':
                val = float(params[2])
                val = max(min(val, MAX_GUIDANCE), 0)

            elif attr == 'upscaler':
                val = params[2]
                if val == 'off':
                    val = None
                elif val != 'x4':
                    raise ConfigUnknownUpscaler(
                        f'\"{val}\" is not a valid upscaler')

            else:
                raise ConfigUnknownAttribute(
                    f'\"{attr}\" not a configurable parameter')

            return attr, val, f'config updated! {attr} to {val}'

        except ValueError:
            raise ValueError(f'\"{val}\" is not a number silly')

