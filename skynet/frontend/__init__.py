#!/usr/bin/python

import json

from typing import Union, Optional
from pathlib import Path
from contextlib import contextmanager as cm

import pynng

from pynng import TLSConfig
from OpenSSL.crypto import (
    load_privatekey,
    load_certificate,
    FILETYPE_PEM
)

from google.protobuf.struct_pb2 import Struct

from ..network import SessionClient
from ..constants import *

from ..protobuf.auth import *
from ..protobuf.skynet_pb2 import SkynetRPCRequest, SkynetRPCResponse


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


@cm
def open_skynet_rpc(
    unique_id: str,
    rpc_address: str = DEFAULT_RPC_ADDR,
    cert_name: Optional[str] = None,
    key_name: Optional[str] = None
):
    sesh = SessionClient(
        rpc_address,
        unique_id,
        cert_name=cert_name,
        key_name=key_name
    )
    logging.debug(f'opening skynet rpc...')
    sesh.connect()
    yield sesh
    sesh.disconnect()

def validate_user_config_request(req: str):
    params = req.split(' ')

    if len(params) < 3:
        raise ConfigRequestFormatError('config request format incorrect')

    else:
        try:
            attr = params[1]

            match attr:
                case 'algo':
                    val = params[2]
                    if val not in ALGOS:
                        raise ConfigUnknownAlgorithm(f'no algo named {val}')

                case 'step':
                    val = int(params[2])
                    val = max(min(val, MAX_STEP), MIN_STEP)

                case 'width':
                    val = max(min(int(params[2]), MAX_WIDTH), 16)
                    if val % 8 != 0:
                        raise ConfigSizeDivisionByEight(
                            'size must be divisible by 8!')

                case 'height':
                    val = max(min(int(params[2]), MAX_HEIGHT), 16)
                    if val % 8 != 0:
                        raise ConfigSizeDivisionByEight(
                            'size must be divisible by 8!')

                case 'seed':
                    val = params[2]
                    if val == 'auto':
                        val = None
                    else:
                        val = int(params[2])

                case 'guidance':
                    val = float(params[2])
                    val = max(min(val, MAX_GUIDANCE), 0)

                case 'strength':
                    val = float(params[2])
                    val = max(min(val, 0.99), 0.01)

                case 'upscaler':
                    val = params[2]
                    if val == 'off':
                        val = None
                    elif val != 'x4':
                        raise ConfigUnknownUpscaler(
                            f'\"{val}\" is not a valid upscaler')

                case _:
                    raise ConfigUnknownAttribute(
                        f'\"{attr}\" not a configurable parameter')

            return attr, val, f'config updated! {attr} to {val}'

        except ValueError:
            raise ValueError(f'\"{val}\" is not a number silly')

