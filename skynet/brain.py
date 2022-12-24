#!/usr/bin/python

import json
import uuid
import base64
import logging
import traceback

from uuid import UUID
from pathlib import Path
from functools import partial
from contextlib import asynccontextmanager as acm
from collections import OrderedDict

import trio
import pynng
import trio_asyncio

from pynng import TLSConfig
from OpenSSL.crypto import (
    load_privatekey,
    load_certificate,
    FILETYPE_PEM
)

from .db import *
from .structs import *
from .constants import *


class SkynetDGPUOffline(BaseException):
    ...

class SkynetDGPUOverloaded(BaseException):
    ...

class SkynetDGPUComputeError(BaseException):
    ...

class SkynetShutdownRequested(BaseException):
    ...


@acm
async def open_rpc_service(sock, dgpu_bus, db_pool, tls_whitelist, tls_key):
    nodes = OrderedDict()
    wip_reqs = {}
    fin_reqs = {}
    next_worker: Optional[int] = None
    security = len(tls_whitelist) > 0

    def connect_node(uid):
        nonlocal next_worker
        nodes[uid] = {
            'task': None
        }
        logging.info(f'dgpu online: {uid}')

        if not next_worker:
            next_worker = 0

    def disconnect_node(uid):
        nonlocal next_worker
        if uid not in nodes:
            return
        i = list(nodes.keys()).index(uid)
        del nodes[uid]

        if i < next_worker:
            next_worker -= 1

        if len(nodes) == 0:
            logging.info('nw: None')
            next_worker = None

        logging.warning(f'dgpu offline: {uid}')

    def is_worker_busy(nid: str):
        return nodes[nid]['task'] != None

    def are_all_workers_busy():
        for nid in nodes.keys():
            if not is_worker_busy(nid):
                return False

        return True

    def get_next_worker():
        nonlocal next_worker
        logging.info('get next_worker called')
        logging.info(f'pre next_worker: {next_worker}')

        if next_worker == None:
            raise SkynetDGPUOffline('No workers connected, try again later')

        if are_all_workers_busy():
            raise SkynetDGPUOverloaded('All workers are busy at the moment')


        nid = list(nodes.keys())[next_worker]
        while is_worker_busy(nid):
            next_worker += 1

            if next_worker >= len(nodes):
                next_worker = 0

            nid = list(nodes.keys())[next_worker]

        next_worker += 1
        if next_worker >= len(nodes):
            next_worker = 0

        logging.info(f'post next_worker: {next_worker}')

        return nid

    async def dgpu_image_streamer():
        nonlocal wip_reqs, fin_reqs
        while True:
            msg = DGPUBusResponse(
                **json.loads(
                    (await dgpu_bus.arecv()).decode()))

            if security:
                msg.verify(tls_whitelist[msg.cert])

            if msg.rid not in wip_reqs:
                continue

            fin_reqs[msg.rid] = msg
            event = wip_reqs[msg.rid]
            event.set()
            del wip_reqs[msg.rid]

    async def dgpu_stream_one_img(req: ImageGenRequest):
        nonlocal wip_reqs, fin_reqs, next_worker
        nid = get_next_worker()
        idx = list(nodes.keys()).index(nid)
        logging.info(f'dgpu_stream_one_img {idx}/{len(nodes)} {nid}')
        rid = uuid.uuid4().hex
        ack_event = trio.Event()
        img_event = trio.Event()
        wip_reqs[rid] = ack_event

        nodes[nid]['task'] = rid

        dgpu_req = DGPUBusRequest(
            rid=rid,
            nid=nid,
            task='diffuse',
            params=req.to_dict())

        logging.info(f'dgpu_bus req: {dgpu_req}')

        if security:
            dgpu_req.sign(tls_key, 'skynet')

        await dgpu_bus.asend(
            json.dumps(dgpu_req.to_dict()).encode())

        with trio.move_on_after(4):
            await ack_event.wait()

        logging.info(f'ack event: {ack_event.is_set()}')

        if not ack_event.is_set():
            disconnect_node(nid)
            raise SkynetDGPUOffline('dgpu failed to acknowledge request')

        ack_msg = fin_reqs[rid]
        if 'ack' not in ack_msg.params:
            disconnect_node(nid)
            raise SkynetDGPUOffline('dgpu failed to acknowledge request')

        wip_reqs[rid] = img_event
        with trio.move_on_after(30):
            await img_event.wait()

        logging.info(f'img event: {ack_event.is_set()}')

        if not img_event.is_set():
            disconnect_node(nid)
            raise SkynetDGPUComputeError('30 seconds timeout while processing request')

        nodes[nid]['task'] = None

        img_resp = fin_reqs[rid]
        del fin_reqs[rid]

        if 'error' in img_resp.params:
            raise SkynetDGPUComputeError(img_resp.params['error'])

        return rid, img_resp.params['img'], img_resp.params['meta']

    async def handle_user_request(rpc_ctx, req):
        try:
            async with db_pool.acquire() as conn:
                user = await get_or_create_user(conn, req.uid)

                result = {}

                match req.method:
                    case 'txt2img':
                        logging.info('txt2img')
                        user_config = {**(await get_user_config(conn, user))}
                        del user_config['id']
                        user_config.update((k, req.params[k]) for k in req.params)
                        req = ImageGenRequest(**user_config)
                        rid, img, meta = await dgpu_stream_one_img(req)
                        logging.info(f'done streaming {rid}')
                        result = {
                            'id': rid,
                            'img': img,
                            'meta': meta
                        }

                        await update_user_stats(conn, user, last_prompt=user_config['prompt'])
                        logging.info('updated user stats.')

                    case 'redo':
                        logging.info('redo')
                        user_config = {**(await get_user_config(conn, user))}
                        del user_config['id']
                        prompt = await get_last_prompt_of(conn, user)

                        if prompt:
                            req = ImageGenRequest(
                                prompt=prompt,
                                **user_config
                            )
                            rid, img, meta = await dgpu_stream_one_img(req)
                            result = {
                                'id': rid,
                                'img': img,
                                'meta': meta
                            }
                            await update_user_stats(conn, user)
                            logging.info('updated user stats.')

                        else:
                            result = {
                                'error': 'skynet_no_last_prompt',
                                'message': 'No prompt to redo, do txt2img first'
                            }

                    case 'config':
                        logging.info('config')
                        if req.params['attr'] in CONFIG_ATTRS:
                            logging.info(f'update: {req.params}')
                            await update_user_config(
                                conn, user, req.params['attr'], req.params['val'])
                            logging.info('done')

                    case 'stats':
                        logging.info('stats')
                        generated, joined, role = await get_user_stats(conn, user)

                        result = {
                            'generated': generated,
                            'joined': joined.strftime(DATE_FORMAT),
                            'role': role
                        }

                    case _:
                        logging.warn('unknown method')

        except SkynetDGPUOffline as e:
            result = {
                'error': 'skynet_dgpu_offline',
                'message': str(e)
            }

        except SkynetDGPUOverloaded as e:
            result = {
                'error': 'skynet_dgpu_overloaded',
                'message': str(e),
                'nodes': len(nodes)
            }

        except SkynetDGPUComputeError as e:
            result = {
                'error': 'skynet_dgpu_compute_error',
                'message': str(e)
            }
        except BaseException as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            result = {
                'error': 'skynet_internal_error',
                'message': str(e)
            }

        resp = SkynetRPCResponse(result=result)

        if security:
            resp.sign(tls_key, 'skynet')

        logging.info('sending response')
        await rpc_ctx.asend(
            json.dumps(resp.to_dict()).encode())
        rpc_ctx.close()
        logging.info('done')

    async def request_service(n):
        nonlocal next_worker
        while True:
            ctx = sock.new_context()
            msg = await ctx.arecv_msg()

            content = msg.bytes.decode()
            req = SkynetRPCRequest(**json.loads(content))

            if security:
                if req.cert not in tls_whitelist:
                    logging.warning(
                        f'{req.cert} not in tls whitelist and security=True')
                    continue

                try:
                    req.verify(tls_whitelist[req.cert])

                except ValueError:
                    logging.warning(
                        f'{req.cert} sent an unauthenticated msg with security=True')
                    continue

            result = {}

            if req.method == 'skynet_shutdown':
                raise SkynetShutdownRequested

            elif req.method == 'dgpu_online':
                connect_node(req.uid)

            elif req.method == 'dgpu_offline':
                disconnect_node(req.uid)

            elif req.method == 'dgpu_workers':
                result = len(nodes)

            elif req.method == 'dgpu_next':
                result = next_worker

            else:
                n.start_soon(
                    handle_user_request, ctx, req)
                continue

            resp = SkynetRPCResponse(
                result={'ok': result})

            if security:
                resp.sign(tls_key, 'skynet')

            await ctx.asend(
                json.dumps(resp.to_dict()).encode())

            ctx.close()


    async with trio.open_nursery() as n:
        n.start_soon(dgpu_image_streamer)
        n.start_soon(request_service, n)
        logging.info('starting rpc service')
        yield
        logging.info('stopping rpc service')
        n.cancel_scope.cancel()


@acm
async def run_skynet(
    db_user: str = DB_USER,
    db_pass: str = DB_PASS,
    db_host: str = DB_HOST,
    rpc_address: str = DEFAULT_RPC_ADDR,
    dgpu_address: str = DEFAULT_DGPU_ADDR,
    security: bool = True
):
    logging.basicConfig(level=logging.INFO)
    logging.info('skynet is starting')

    tls_config = None
    if security:
        # load tls certs
        certs_dir = Path(DEFAULT_CERTS_DIR).resolve()

        tls_key_data = (certs_dir / DEFAULT_CERT_SKYNET_PRIV).read_text()
        tls_key = load_privatekey(FILETYPE_PEM, tls_key_data)

        tls_cert_data = (certs_dir / DEFAULT_CERT_SKYNET_PUB).read_text()
        tls_cert = load_certificate(FILETYPE_PEM, tls_cert_data)

        tls_whitelist = {}
        for cert_path in (certs_dir / 'whitelist').glob('*.cert'):
            tls_whitelist[cert_path.stem] = load_certificate(
                FILETYPE_PEM, cert_path.read_text())

        cert_start = tls_cert_data.index('\n') + 1
        logging.info(f'tls_cert: {tls_cert_data[cert_start:cert_start+64]}...')
        logging.info(f'tls_whitelist len: {len(tls_whitelist)}')

        rpc_address = 'tls+' + rpc_address
        dgpu_address = 'tls+' + dgpu_address
        tls_config = TLSConfig(
            TLSConfig.MODE_SERVER,
            own_key_string=tls_key_data,
            own_cert_string=tls_cert_data)

    with (
        pynng.Rep0(recv_max_size=0) as rpc_sock,
        pynng.Bus0(recv_max_size=0) as dgpu_bus
    ):
        async with open_database_connection(
            db_user, db_pass, db_host) as db_pool:

            logging.info('connected to db.')
            if security:
                rpc_sock.tls_config = tls_config
                dgpu_bus.tls_config = tls_config

            rpc_sock.listen(rpc_address)
            dgpu_bus.listen(dgpu_address)

            try:
                async with open_rpc_service(
                    rpc_sock, dgpu_bus, db_pool, tls_whitelist, tls_key):
                    yield

            except SkynetShutdownRequested:
                ...

        logging.info('disconnected from db.')
