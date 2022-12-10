#!/usr/bin/python

import json

from typing import Union
from contextlib import contextmanager as cm

import pynng

from ..types import SkynetRPCRequest, SkynetRPCResponse
from ..constants import DEFAULT_RPC_ADDR


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


@cm
def open_skynet_rpc(rpc_address: str = DEFAULT_RPC_ADDR):
    with pynng.Req0(dial=rpc_address) as rpc_sock:
        yield rpc_sock
