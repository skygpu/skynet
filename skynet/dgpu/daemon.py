import os
import logging
import contextlib
from functools import partial

import trio
import msgspec

from skynet.config import DgpuConfig as Config
from skynet.types import (
    BodyV0
)
from skynet.contract import GPUContractAPI
from skynet.constants import MODELS
from skynet.ipfs import AsyncIPFSHTTP, get_ipfs_img
from skynet.dgpu.errors import DGPUComputeError
from skynet.dgpu.tui import maybe_update_tui, maybe_update_tui_async
from skynet.dgpu.compute import maybe_load_model, compute_one
from skynet.dgpu.network import (
    ContractState,
)


async def maybe_update_tui_balance(contract: GPUContractAPI):
    async def _fn(tui):
        # update balance
        balance = (await contract.get_user(tui.config.account)).balance
        tui.set_header_text(new_balance=f'balance: {balance}')

    await maybe_update_tui_async(_fn)


async def maybe_serve_one(
    config: Config,
    contract: GPUContractAPI,
    ipfs_api: AsyncIPFSHTTP,
    state_mngr: ContractState,
):
    logging.info(f'maybe serve request pi: {state_mngr.poll_index}')
    req = state_mngr.first

    # no requests in queue
    if not req:
        return

    # parse request
    body = msgspec.json.decode(req.body, type=BodyV0)
    model = body.params.model

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
        logging.warning('model is blacklisted!, skip...')
        return

    # if worker already produced a result for this request
    if state_mngr.is_request_filled(req.id):
        logging.info(f'worker already submitted a result for request #{req.id}, skip...')
        return

    # skip if workers in non_compete already on it
    if not state_mngr.should_compete_for_id(req.id):
        logging.info('worker in configured non_compete list already working on request, skip...')
        return

    # resolve the ipfs hashes into the actual data behind them
    inputs = []
    raw_inputs = req.binary_data.split(',')
    if raw_inputs:
        logging.info(f'fetching IPFS inputs: {raw_inputs}')

    retry = 3
    for _input in raw_inputs:
        if _input:
            for r in range(retry):
                try:
                    # user `GPUConnector` to IO with
                    # storage layer to seed the compute
                    # task.
                    img = await get_ipfs_img(f'https://{config.ipfs_domain}/ipfs/{_input}')
                    inputs.append(img)
                    logging.info(f'retrieved {_input}!')
                    break

                except BaseException:
                    logging.exception(
                        f'IPFS fetch input error !?! retries left {retry - r - 1}\n'
                    )

    total_step = body.params.step
    mode = body.method

    # TODO: validate request

    resp = await contract.accept_work(config.account, req.id)
    if not resp or 'code' in resp:
        logging.info('begin_work error, probably being worked on already... skip.')
        return

    with maybe_load_model(model, mode) as model:
        try:
            maybe_update_tui(lambda tui: tui.set_progress(0, done=total_step))

            output_type = body.params.output_type
            output = None
            output_hash = None
            match config.backend:
                case 'sync-on-thread':
                    '''Block this task until inference completes, pass
                    state_mngr.should_cancel_work predicate as the inference_step_wakeup cb
                    used by torch each step of the inference, it will use a
                    trio.from_thread to unblock the main thread and pump the event loop
                    '''
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            output_hash, output = await trio.to_thread.run_sync(
                                partial(
                                    compute_one,
                                    model,
                                    req.id,
                                    mode, body.params,
                                    inputs=inputs,
                                    should_cancel=state_mngr.should_cancel_work,
                                )
                            )

                case _:
                    raise DGPUComputeError(
                        f'Unsupported backend {config.backend}'
                    )

            maybe_update_tui(lambda tui: tui.set_progress(total_step))

            ipfs_hash = await ipfs_api.publish(output, type=output_type)

            maybe_request_hash = None
            if config.proto_version == 0:
                maybe_request_hash = req.hash_v0()

            await contract.submit_work(
                config.account, req.id, output_hash, ipfs_hash, request_hash=maybe_request_hash)

            await state_mngr.update_state()

            await maybe_update_tui_balance(contract)

            await state_mngr.update_state()


        except BaseException as err:
            if 'network cancel' not in str(err):
                logging.exception('Failed to serve model request !?\n')

            await state_mngr.update_state()

            if state_mngr.is_request_in_progress(req.id):
                await contract.cancel_work(config.account, req.id, 'reason not provided')


async def dgpu_serve_forever(
    config: Config,
    contract: GPUContractAPI,
    ipfs_api: AsyncIPFSHTTP,
    state_mngr: ContractState
):
    await maybe_update_tui_balance(contract)
    maybe_update_tui(lambda tui: tui.set_header_text(new_worker_name=config.account))

    last_poll_idx = -1
    try:
        while True:
            await state_mngr.wait_data_update()
            if state_mngr.poll_index == last_poll_idx:
                await trio.sleep(config.poll_time)
                continue

            last_poll_idx = state_mngr.poll_index

            await maybe_serve_one(config, contract, ipfs_api, state_mngr)

    except KeyboardInterrupt:
        ...
