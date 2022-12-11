#!/usr/bin/python

import json
import uuid
import base64
import logging

from uuid import UUID
from functools import partial
from collections import OrderedDict

import trio
import pynng
import trio_asyncio

from .db import *
from .types import *
from .constants import *


class SkynetDGPUOffline(BaseException):
    ...

class SkynetDGPUOverloaded(BaseException):
    ...


async def rpc_service(sock, dgpu_bus, db_pool):
    nodes = OrderedDict()
    wip_reqs = {}
    fin_reqs = {}

    def is_worker_busy(nid: int):
        for task in nodes[nid]['tasks']:
            if task != None:
                return False

        return True

    def are_all_workers_busy():
        for nid in nodes.keys():
            if not is_worker_busy(nid):
                return False

        return True

    next_worker: Optional[int] = None
    def get_next_worker():
        nonlocal next_worker

        if not next_worker:
            raise SkynetDGPUOffline

        if are_all_workers_busy():
            raise SkynetDGPUOverloaded

        while is_worker_busy(next_worker):
            next_worker += 1

            if next_worker >= len(nodes):
                next_worker = 0

        return next_worker

    async def dgpu_image_streamer():
        nonlocal wip_reqs, fin_reqs
        while True:
            msg = await dgpu_bus.arecv_msg()
            rid = UUID(bytes=msg.bytes[:16]).hex
            img = msg.bytes[16:].hex()
            fin_reqs[rid] = img
            event = wip_reqs[rid]
            event.set()
            del wip_reqs[rid]

    async def dgpu_stream_one_img(req: ImageGenRequest):
        nonlocal wip_reqs, fin_reqs, next_worker
        nid = get_next_worker()
        logging.info(f'dgpu_stream_one_img {next_worker} {nid}')
        rid = uuid.uuid4().hex
        event = trio.Event()
        wip_reqs[rid] = event

        tid = nodes[nid]['tasks'].index(None)
        nodes[nid]['tasks'][tid] = rid

        dgpu_req = DGPUBusRequest(
            rid=rid,
            nid=nid,
            task='diffuse',
            params=req.to_dict())

        logging.info(f'dgpu_bus req: {dgpu_req}')

        await dgpu_bus.asend(
            json.dumps(dgpu_req.to_dict()).encode())

        await event.wait()

        nodes[nid]['tasks'][tid] = None

        img = fin_reqs[rid]
        del fin_reqs[rid]

        logging.info(f'done streaming {img}')

        return rid, img

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
                        prompt = req.params['prompt']
                        req = ImageGenRequest(
                            prompt=prompt,
                            **user_config
                        )
                        rid, img = await dgpu_stream_one_img(req)
                        result = {
                            'id': rid,
                            'img': img
                        }

                    case 'redo':
                        logging.info('redo')
                        user_config = await get_user_config(conn, user)
                        prompt = await get_last_prompt_of(conn, user)
                        req = ImageGenRequest(
                            prompt=prompt,
                            **user_config
                        )
                        rid, img = await dgpu_stream_one_img(req)
                        result = {
                            'id': rid,
                            'img': img
                        }

                    case 'config':
                        logging.info('config')
                        if req.params['attr'] in CONFIG_ATTRS:
                            await update_user_config(
                                conn, user, req.params['attr'], req.params['val'])

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

        except SkynetDGPUOffline:
            result = {
                'error': 'skynet_dgpu_offline'
            }

        except SkynetDGPUOverloaded:
            result = {
                'error': 'skynet_dgpu_overloaded',
                'nodes': len(nodes)
            }

        except BaseException as e:
            logging.error(e)
            result = {
                'error': 'skynet_internal_error'
            }

        await rpc_ctx.asend(
            json.dumps(
                SkynetRPCResponse(result=result).to_dict()).encode())


    async with trio.open_nursery() as n:
        n.start_soon(dgpu_image_streamer)
        while True:
            ctx = sock.new_context()
            msg = await ctx.arecv_msg()
            content = msg.bytes.decode()
            req = SkynetRPCRequest(**json.loads(content))

            logging.info(req)

            result = {}

            if req.method == 'dgpu_online':
                nodes[req.uid] = {
                    'tasks': [None for _ in range(req.params['max_tasks'])],
                    'max_tasks': req.params['max_tasks']
                }
                logging.info(f'dgpu online: {req.uid}')

                if not next_worker:
                    next_worker = 0

            elif req.method == 'dgpu_offline':
                i = list(nodes.keys()).index(req.uid)
                del nodes[req.uid]

                if i < next_worker:
                    next_worker -= 1

                if len(nodes) == 0:
                    next_worker = None

                logging.info(f'dgpu offline: {req.uid}')

            elif req.method == 'dgpu_workers':
                result = len(nodes)

            elif req.method == 'dgpu_next':
                result = next_worker

            else:
                n.start_soon(
                    handle_user_request, ctx, req)
                continue

            await ctx.asend(
                json.dumps(
                    SkynetRPCResponse(
                        result={'ok': result}).to_dict()).encode())


async def run_skynet(
    db_user: str = DB_USER,
    db_pass: str = DB_PASS,
    db_host: str = DB_HOST,
    rpc_address: str = DEFAULT_RPC_ADDR,
    dgpu_address: str = DEFAULT_DGPU_ADDR,
    task_status = trio.TASK_STATUS_IGNORED
):
    logging.basicConfig(level=logging.INFO)
    logging.info('skynet is starting')

    async with (
        trio.open_nursery() as n,
        open_database_connection(
            db_user, db_pass, db_host) as db_pool
    ):
        logging.info('connected to db.')
        with (
            pynng.Rep0(listen=rpc_address) as rpc_sock,
            pynng.Bus0(listen=dgpu_address) as dgpu_bus
        ):
            n.start_soon(
                rpc_service, rpc_sock, dgpu_bus, db_pool)
            task_status.started()

            try:
                await trio.sleep_forever()

            except KeyboardInterrupt:
                ...

