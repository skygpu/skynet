#!/usr/bin/python

import trio
import json
import uuid
import logging

import pynng
import tractor

from . import gpu
from .gpu import open_gpu_worker
from .types import *
from .constants import *
from .frontend import rpc_call


async def open_dgpu_node(
    rpc_address: str = DEFAULT_RPC_ADDR,
    dgpu_address: str = DEFAULT_DGPU_ADDR,
    dgpu_max_tasks: int = DEFAULT_DGPU_MAX_TASKS,
    initial_algos: str = DEFAULT_INITAL_ALGOS
):
    logging.basicConfig(level=logging.INFO)

    name = uuid.uuid4()
    workers = initial_algos.copy()
    tasks = [None for _ in range(dgpu_max_tasks)]

    portal_map: dict[int, tractor.Portal]
    contexts: dict[int, tractor.Context]

    def get_next_worker(need_algo: str):
        nonlocal workers, tasks
        for task, algo in zip(workers, tasks):
            if need_algo == algo and not task:
                return workers.index(need_algo)

        return tasks.index(None)

    async def gpu_streamer(
        ctx: tractor.Context,
        nid: int
    ):
        nonlocal tasks
        async with ctx.open_stream() as stream:
            async for img in stream:
                tasks[nid]['res'] = img
                tasks[nid]['event'].set()

    async def gpu_compute_one(ireq: ImageGenRequest):
        wid = get_next_worker(ireq.algo)
        event = trio.Event()

        workers[wid] = ireq.algo
        tasks[wid] = {
            'res': None, 'event': event}

        await contexts[i].send(ireq)

        await event.wait()

        img = tasks[wid]['res']
        tasks[wid] = None
        return img


    with (
        pynng.Req0(dial=rpc_address) as rpc_sock,
        pynng.Bus0(dial=dgpu_address) as dgpu_sock
    ):
        async def _rpc_call(*args, **kwargs):
            return await rpc_call(rpc_sock, *args, **kwargs)

        async def _process_dgpu_req(req: DGPUBusRequest):
            img = await gpu_compute_one(
                ImageGenRequest(**req.params))
            await dgpu_sock.asend(
                bytes.fromhex(req.rid) + img)

        res = await _rpc_call(
            name.hex, 'dgpu_online', {'max_tasks': dgpu_max_tasks})
        logging.info(res)
        assert 'ok' in res.result

        async with (
            tractor.open_actor_cluster(
                modules=['skynet_bot.gpu'],
                count=dgpu_max_tasks,
                names=[i for i in range(dgpu_max_tasks)]
                ) as portal_map,
            trio.open_nursery() as n
        ):
            logging.info(f'starting {dgpu_max_tasks} gpu workers')
            async with tractor.gather_contexts((
                ctx.open_context(
                    open_gpu_worker, algo, 1.0 / dgpu_max_tasks)
            )) as contexts:
                contexts = {i: ctx for i, ctx in enumerate(contexts)}
                for i, ctx in contexts.items():
                    n.start_soon(
                        gpu_streamer, ctx, i)
                try:
                    while True:
                        msg = await dgpu_sock.arecv()
                        req = DGPUBusRequest(
                            **json.loads(msg.decode()))

                        if req.nid != name.hex:
                            continue

                        logging.info(f'dgpu: {name}, req: {req}')
                        n.start_soon(
                            _process_dgpu_req, req)

                except KeyboardInterrupt:
                    ...

        res = await _rpc_call(name.hex, 'dgpu_offline')
        logging.info(res)
        assert 'ok' in res.result
