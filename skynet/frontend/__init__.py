#!/usr/bin/python

import json

from typing import Union, Optional
from pathlib import Path
from contextlib import asynccontextmanager as acm

import pynng

from pynng import TLSConfig
from OpenSSL.crypto import (
    load_privatekey,
    load_certificate,
    FILETYPE_PEM
)

from ..structs import SkynetRPCRequest, SkynetRPCResponse
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


@acm
async def open_skynet_rpc(
    unique_id: str,
    rpc_address: str = DEFAULT_RPC_ADDR,
    security: bool = False,
    cert_name: Optional[str] = None,
    key_name: Optional[str] = None
):
    tls_config = None

    if security:
        # load tls certs
        if not key_name:
            key_name = cert_name

        certs_dir = Path(DEFAULT_CERTS_DIR).resolve()

        skynet_cert_data = (certs_dir / 'brain.cert').read_text()
        skynet_cert = load_certificate(FILETYPE_PEM, skynet_cert_data)

        tls_cert_path = certs_dir / f'{cert_name}.cert'
        tls_cert_data = tls_cert_path.read_text()
        tls_cert = load_certificate(FILETYPE_PEM, tls_cert_data)
        cert_name = tls_cert_path.stem

        tls_key_data = (certs_dir / f'{key_name}.key').read_text()
        tls_key = load_privatekey(FILETYPE_PEM, tls_key_data)

        rpc_address = 'tls+' + rpc_address
        tls_config = TLSConfig(
            TLSConfig.MODE_CLIENT,
            own_key_string=tls_key_data,
            own_cert_string=tls_cert_data,
            ca_string=skynet_cert_data)

    with pynng.Req0(recv_max_size=0) as sock:
        if security:
            sock.tls_config = tls_config

        sock.dial(rpc_address)

        async def _rpc_call(
            method: str,
            params: dict = {},
            uid: Optional[Union[int, str]] = None
        ):
            req = SkynetRPCRequest(
                uid=uid if uid else unique_id,
                method=method,
                params=params
            )

            if security:
                req.sign(tls_key, cert_name)

            ctx = sock.new_context()
            await ctx.asend(
                json.dumps(
                    req.to_dict()).encode())

            resp = SkynetRPCResponse(
                **json.loads((await ctx.arecv()).decode()))
            ctx.close()

            if security:
                resp.verify(skynet_cert)

            return resp

        yield _rpc_call


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

