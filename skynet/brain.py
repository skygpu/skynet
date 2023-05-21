#!/usr/bin/python

import logging

from contextlib import asynccontextmanager as acm
from collections import OrderedDict

import trio

from pynng import Context

from .utils import time_ms
from .network import *
from .protobuf import *
from .constants import *



class SkynetRPCBadRequest(BaseException):
    ...

class SkynetDGPUOffline(BaseException):
    ...

class SkynetDGPUOverloaded(BaseException):
    ...

class SkynetDGPUComputeError(BaseException):
    ...

class SkynetShutdownRequested(BaseException):
    ...


@acm
async def run_skynet(
    rpc_address: str = DEFAULT_RPC_ADDR
):
    logging.basicConfig(level=logging.INFO)
    logging.info('skynet is starting')

    nodes = OrderedDict()
    heartbeats = {}
    next_worker: Optional[int] = None

    def connect_node(req: SkynetRPCRequest):
        nonlocal next_worker

        node_params = MessageToDict(req.params)
        logging.info(f'got node params {node_params}')

        if 'dgpu_addr' not in node_params:
            raise SkynetRPCBadRequest(
                f'DGPU connection params don\'t include dgpu addr')

        session = SessionClient(
            node_params['dgpu_addr'],
            'skynet',
            cert_name='brain.cert',
            key_name='brain.key',
            ca_name=node_params['cert']
        )
        try:
            session.connect()

            node = {
                'task': None,
                'session': session
            }
            node.update(node_params)

            nodes[req.uid] = node
            logging.info(f'DGPU node online: {req.uid}')

            if not next_worker:
                next_worker = 0

        except pynng.exceptions.ConnectionRefused:
            logging.warning(f'error while dialing dgpu node... dropping...')
            raise SkynetDGPUOffline('Connection to dgpu node addr failed.')

    def disconnect_node(uid):
        nonlocal next_worker
        if uid not in nodes:
            logging.warning(f'Attempt to disconnect unknown node {uid}')
            return

        i = list(nodes.keys()).index(uid)
        nodes[uid]['session'].disconnect()
        del nodes[uid]

        if i < next_worker:
            next_worker -= 1

        logging.warning(f'DGPU node offline: {uid}')

        if len(nodes) == 0:
            logging.info('All nodes disconnected.')
            next_worker = None


    def is_worker_busy(nid: str):
        return nodes[nid]['task'] != None

    def are_all_workers_busy():
        for nid in nodes.keys():
            if not is_worker_busy(nid):
                return False

        return True

    def get_next_worker():
        nonlocal next_worker

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

        return nid

    async def rpc_handler(req: SkynetRPCRequest, ctx: Context):
        result = {'ok': {}}
        resp = SkynetRPCResponse()

        try:
            match req.method:
                case 'dgpu_online':
                    connect_node(req)

                case 'dgpu_call':
                    nid = get_next_worker()
                    idx = list(nodes.keys()).index(nid)
                    node = nodes[nid]
                    logging.info(f'dgpu_call {idx}/{len(nodes)} {nid} @ {node["dgpu_addr"]}')
                    dgpu_time = await node['session'].rpc('dgpu_time')
                    if 'ok' not in dgpu_time.result:
                        status = MessageToDict(dgpu_time.result)
                        logging.warning(json.dumps(status, indent=4))
                        disconnect_node(nid)
                        raise SkynetDGPUComputeError(status['error'])

                    dgpu_time = dgpu_time.result['ok']
                    logging.info(f'ping to {nid}: {time_ms() - dgpu_time} ms')

                    try:
                        dgpu_result = await node['session'].rpc(
                            timeout=45,  # give this 45 sec to run cause its compute
                            binext=req.bin,
                            **req.params
                        )
                        result = MessageToDict(dgpu_result.result)

                        if dgpu_result.bin:
                            resp.bin = dgpu_result.bin

                    except trio.TooSlowError:
                        result = {'error': 'timeout while processing request'}

                case 'dgpu_offline':
                    disconnect_node(req.uid)

                case 'dgpu_workers':
                    result = {'ok': len(nodes)}

                case 'dgpu_next':
                    result = {'ok': next_worker}

                case 'skynet_shutdown':
                    raise SkynetShutdownRequested

                case _:
                    logging.warning(f'Unknown method {req.method}')
                    result = {'error': 'unknown method'}

        except BaseException as e:
            result = {'error': str(e)}

        resp.result.update(result)

        return resp

    rpc_server = SessionServer(
        rpc_address,
        rpc_handler,
        cert_name='brain.cert',
        key_name='brain.key'
    )

    async with rpc_server.open():
        logging.info('rpc server is up')
        yield
        logging.info('skynet is shuting down...')

    logging.info('skynet down.')
