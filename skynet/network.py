#!/usr/bin/python

import zlib
import socket

from typing import Callable, Awaitable, Optional
from pathlib import Path
from contextlib import asynccontextmanager as acm
from cryptography import x509
from cryptography.hazmat.primitives import serialization

import trio
import pynng

from pynng import TLSConfig, Context

from .protobuf import *
from .constants import *


def get_random_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 0))
    return s.getsockname()[1]


def load_certs(
    certs_dir: str,
    cert_name: str,
    key_name: str
):
    certs_dir = Path(certs_dir).resolve()
    tls_key_data = (certs_dir / key_name).read_bytes()
    tls_key = serialization.load_pem_private_key(
        tls_key_data,
        password=None
    )

    tls_cert_data = (certs_dir / cert_name).read_bytes()
    tls_cert = x509.load_pem_x509_certificate(
        tls_cert_data
    )

    tls_whitelist = {}
    for cert_path in (*(certs_dir / 'whitelist').glob('*.cert'), certs_dir / 'brain.cert'):
        tls_whitelist[cert_path.stem] = x509.load_pem_x509_certificate(
            cert_path.read_bytes()
        )

    return (
        SessionTLSConfig(
            TLSConfig.MODE_SERVER,
            own_key_string=tls_key_data,
            own_cert_string=tls_cert_data
        ),

        tls_whitelist
    )


def load_certs_client(
    certs_dir: str,
    cert_name: str,
    key_name: str,
    ca_name: Optional[str] = None
):
    certs_dir = Path(certs_dir).resolve()
    if not ca_name:
        ca_name = 'brain.cert'

    ca_cert_data = (certs_dir / ca_name).read_bytes()

    tls_key_data = (certs_dir / key_name).read_bytes()


    tls_cert_data = (certs_dir / cert_name).read_bytes()


    tls_whitelist = {}
    for cert_path in (*(certs_dir / 'whitelist').glob('*.cert'), certs_dir / 'brain.cert'):
        tls_whitelist[cert_path.stem] = x509.load_pem_x509_certificate(
            cert_path.read_bytes()
        )

    return (
        SessionTLSConfig(
            TLSConfig.MODE_CLIENT,
            own_key_string=tls_key_data,
            own_cert_string=tls_cert_data,
            ca_string=ca_cert_data
        ),

        tls_whitelist
    )


class SessionError(BaseException):
    ...


class SessionTLSConfig(TLSConfig):

    def __init__(
        self,
        mode,
        server_name=None,
        ca_string=None,
        own_key_string=None,
        own_cert_string=None,
        auth_mode=None,
        ca_files=None,
        cert_key_file=None,
        passwd=None
    ):
        super().__init__(
            mode,
            server_name=server_name,
            ca_string=ca_string,
            own_key_string=own_key_string,
            own_cert_string=own_cert_string,
            auth_mode=auth_mode,
            ca_files=ca_files,
            cert_key_file=cert_key_file,
            passwd=passwd
        )

        if ca_string:
            self.ca_cert = x509.load_pem_x509_certificate(ca_string)

        self.cert = x509.load_pem_x509_certificate(own_cert_string)
        self.key = serialization.load_pem_private_key(
            own_key_string,
            password=passwd
        )


class SessionServer:

    def __init__(
        self,
        addr: str,
        msg_handler: Callable[
            [SkynetRPCRequest, Context], Awaitable[SkynetRPCResponse]
        ],
        cert_name: Optional[str] = None,
        key_name: Optional[str] = None,
        cert_dir: str = DEFAULT_CERTS_DIR,
        recv_max_size = 0
    ):
        self.addr = addr
        self.msg_handler = msg_handler

        self.cert_name = cert_name
        self.tls_config = None
        self.tls_whitelist = None
        if cert_name and key_name:
            self.cert_name = cert_name
            self.tls_config, self.tls_whitelist = load_certs(
                cert_dir, cert_name, key_name)

            self.addr = 'tls+' + self.addr

        self.recv_max_size = recv_max_size

    async def _handle_msg(self, req: SkynetRPCRequest, ctx: Context):
        resp = await self.msg_handler(req, ctx)

        if self.tls_config:
            resp.auth.cert = 'skynet'
            resp.auth.sig = sign_protobuf_msg(
                resp, self.tls_config.key)

        raw_msg = zlib.compress(resp.SerializeToString())

        await ctx.asend(raw_msg)

        ctx.close()

    async def _listener (self, sock):
        async with trio.open_nursery() as n:
            while True:
                ctx = sock.new_context()

                raw_msg = await ctx.arecv()
                raw_size = len(raw_msg)
                logging.debug(f'rpc server new msg {raw_size} bytes')

                try:
                    msg = zlib.decompress(raw_msg)
                    msg_size = len(msg)

                except zlib.error:
                    logging.warning(f'Zlib decompress error, dropping msg of size {len(raw_msg)}')
                    continue

                logging.debug(f'msg after decompress {msg_size} bytes, +{msg_size - raw_size} bytes')

                req = SkynetRPCRequest()
                try:
                    req.ParseFromString(msg)

                except google.protobuf.message.DecodeError:
                    logging.warning(f'Dropping malfomed msg of size {len(msg)}')
                    continue

                logging.debug(f'msg method: {req.method}')

                if self.tls_config:
                    if req.auth.cert not in self.tls_whitelist:
                        logging.warning(
                            f'{req.auth.cert} not in tls whitelist')
                        continue

                    try:
                        verify_protobuf_msg(req, self.tls_whitelist[req.auth.cert])

                    except ValueError:
                        logging.warning(
                            f'{req.cert} sent an unauthenticated msg')
                        continue

                n.start_soon(self._handle_msg, req, ctx)

    @acm
    async def open(self):
        with pynng.Rep0(
            recv_max_size=self.recv_max_size
        ) as sock:

            if self.tls_config:
                sock.tls_config = self.tls_config

            sock.listen(self.addr)

            logging.debug(f'server socket listening at {self.addr}')

            async with trio.open_nursery() as n:
                n.start_soon(self._listener, sock)

                try:
                    yield self

                finally:
                    n.cancel_scope.cancel()

        logging.debug('server socket is off.')


class SessionClient:

    def __init__(
        self,
        connect_addr: str,
        uid: str,
        cert_name: Optional[str] = None,
        key_name: Optional[str] = None,
        ca_name: Optional[str] = None,
        cert_dir: str = DEFAULT_CERTS_DIR,
        recv_max_size = 0
    ):
        self.uid = uid
        self.connect_addr = connect_addr

        self.cert_name = None
        self.tls_config = None
        self.tls_whitelist = None
        self.tls_cert = None
        self.tls_key = None
        if cert_name and key_name:
            self.cert_name = Path(cert_name).stem
            self.tls_config, self.tls_whitelist = load_certs_client(
                cert_dir, cert_name, key_name, ca_name=ca_name)

            if not self.connect_addr.startswith('tls'):
                self.connect_addr = 'tls+' + self.connect_addr

        self.recv_max_size = recv_max_size

        self._connected = False
        self._sock = None

    def connect(self):
        self._sock = pynng.Req0(
            recv_max_size=0,
            name=self.uid
        )

        if self.tls_config:
            self._sock.tls_config = self.tls_config

        logging.debug(f'client is dialing {self.connect_addr}...')
        self._sock.dial(self.connect_addr, block=True)
        self._connected = True
        logging.debug(f'client is connected to {self.connect_addr}')

    def disconnect(self):
        self._sock.close()
        self._connected = False
        logging.debug(f'client disconnected.')

    async def rpc(
        self,
        method: str,
        params: dict = {},
        binext: Optional[bytes] = None,
        timeout: float = 2.
    ):
        if not self._connected:
            raise SessionError('tried to use rpc without connecting')

        req = SkynetRPCRequest()
        req.uid = self.uid
        req.method = method
        req.params.update(params)
        if binext:
            logging.debug('added binary extension')
            req.bin = binext

        if self.tls_config:
            req.auth.cert = self.cert_name
            req.auth.sig = sign_protobuf_msg(req, self.tls_config.key)

        with trio.fail_after(timeout):
            ctx = self._sock.new_context()
            raw_req = zlib.compress(req.SerializeToString())
            logging.debug(f'rpc client sending new msg {method} of size {len(raw_req)}')
            await ctx.asend(raw_req)
            logging.debug('sent, awaiting response...')
            raw_resp = await ctx.arecv()
            logging.debug(f'rpc client got response of size {len(raw_resp)}')
            raw_resp = zlib.decompress(raw_resp)

            resp = SkynetRPCResponse()
            resp.ParseFromString(raw_resp)
            ctx.close()

            if self.tls_config:
                verify_protobuf_msg(resp, self.tls_config.ca_cert)

            return resp
