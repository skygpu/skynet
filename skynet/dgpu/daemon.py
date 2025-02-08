import json
import logging
import random
import time
from datetime import datetime
from functools import partial
from hashlib import sha256

import trio

from skynet.config import DgpuConfig as Config
from skynet.constants import (
    MODELS,
    VERSION,
)
from skynet.dgpu.errors import (
    DGPUComputeError,
)
from skynet.dgpu.tui import maybe_update_tui, maybe_update_tui_async
from skynet.dgpu.compute import maybe_load_model, compute_one
from skynet.dgpu.network import NetConnector


def convert_reward_to_int(reward_str):
    int_part, decimal_part = (
        reward_str.split('.')[0],
        reward_str.split('.')[1].split(' ')[0]
    )
    return int(int_part + decimal_part)


async def maybe_update_tui_balance(conn: NetConnector):
    async def _fn(tui):
        # update balance
        balance = await conn.get_worker_balance()
        tui.set_header_text(new_balance=f'balance: {balance}')

    await maybe_update_tui_async(_fn)


async def maybe_serve_one(
    config: Config,
    conn: NetConnector,
    req: dict,
):
    rid = req['id']
    logging.info(f'maybe serve request #{rid}')

    # parse request
    body = json.loads(req['body'])
    model = body['params']['model']

    # if model not known, ignore.
    if model not in MODELS:
        logging.warning(f'unknown model {model}!, skip...')
        return

    # only handle whitelisted models
    if (
        len(config.model_whitelist) > 0
        and
        model not in config.model_whitelist
    ):
        logging.warning('model not whitelisted!, skip...')
        return

    # if blacklist contains model skip
    if (
        len(config.model_blacklist) > 0
        and
        model in config.model_blacklist
    ):
        logging.warning('model not blacklisted!, skip...')
        return

    results = [res['request_id'] for res in conn._tables['results']]

    # if worker already produced a result for this request
    if rid in results:
        logging.info(f'worker already submitted a result for request #{rid}, skip...')
        return

    statuses = conn._tables['requests'][rid]

    # skip if workers in non_compete already on it
    competitors = set((status['worker'] for status in statuses))
    if bool(config.non_compete & competitors):
        logging.info('worker in configured non_compete list already working on request, skip...')
        return

    # resolve the ipfs hashes into the actual data behind them
    inputs = []
    raw_inputs = req['binary_data'].split(',')
    if raw_inputs:
        logging.info(f'fetching IPFS inputs: {raw_inputs}')

    retry = 3
    for _input in req['binary_data'].split(','):
        if _input:
            for r in range(retry):
                try:
                    # user `GPUConnector` to IO with
                    # storage layer to seed the compute
                    # task.
                    img = await conn.get_input_data(_input)
                    inputs.append(img)
                    logging.info(f'retrieved {_input}!')
                    break

                except BaseException:
                    logging.exception(
                        f'IPFS fetch input error !?! retries left {retry - r - 1}\n'
                    )

    # compute unique request hash used on submit
    hash_str = (
        str(req['nonce'])
        +
        req['body']
        +
        req['binary_data']
    )
    logging.debug(f'hashing: {hash_str}')
    request_hash = sha256(hash_str.encode('utf-8')).hexdigest()
    logging.info(f'calculated request hash: {request_hash}')

    total_step = body['params']['step']
    model = body['params']['model']
    mode = body['method']

    # TODO: validate request

    resp = await conn.begin_work(rid)
    if not resp or 'code' in resp:
        logging.info('begin_work error, probably being worked on already... skip.')
        return

    with maybe_load_model(model, mode) as model:
        try:
            maybe_update_tui(lambda tui: tui.set_progress(0, done=total_step))

            output_type = 'png'
            if 'output_type' in body['params']:
                output_type = body['params']['output_type']

            output = None
            output_hash = None
            match config.backend:
                case 'sync-on-thread':
                    output_hash, output = await trio.to_thread.run_sync(
                        partial(
                            compute_one,
                            model,
                            rid,
                            mode, body['params'],
                            inputs=inputs,
                            should_cancel=conn.should_cancel_work,
                        )
                    )

                case _:
                    raise DGPUComputeError(
                        f'Unsupported backend {config.backend}'
                    )

            maybe_update_tui(lambda tui: tui.set_progress(total_step))

            ipfs_hash = await conn.publish_on_ipfs(output, typ=output_type)

            await conn.submit_work(rid, request_hash, output_hash, ipfs_hash)

            await maybe_update_tui_balance(conn)


        except BaseException as err:
            if 'network cancel' not in str(err):
                logging.exception('Failed to serve model request !?\n')

            if rid in conn._tables['requests']:
                await conn.cancel_work(rid, 'reason not provided')


async def dgpu_serve_forever(config: Config, conn: NetConnector):
    await maybe_update_tui_balance(conn)

    try:
        while True:
            await conn.wait_data_update()
            queue = conn._tables['queue']

            random.shuffle(queue)
            queue = sorted(
                queue,
                key=lambda req: convert_reward_to_int(req['reward']),
                reverse=True
            )

            if len(queue) > 0:
                await maybe_serve_one(config, conn, queue[0])

    except KeyboardInterrupt:
        ...
