import json
import time
import random
import logging
from contextlib import asynccontextmanager as acm
from functools import partial

import trio
import leap
import anyio
import httpx
import outcome
import msgspec
from skynet.dgpu.tui import maybe_update_tui
from skynet.config import load_skynet_toml
from skynet.contract import GPUContractAPI
from skynet.types import (
    BodyV0,
    Request,
    WorkerStatusV0,
    ResultV0
)


REQUEST_UPDATE_TIME: int = 3


async def failable(fn: partial, ret_fail=None):
    o = await outcome.acapture(fn)
    match o:
        case outcome.Error(error=(
            OSError() |
            json.JSONDecodeError() |
            anyio.BrokenResourceError() |
            httpx.ConnectError() |
            httpx.ConnectTimeout() |
            httpx.ReadError() |
            httpx.ReadTimeout() |
            leap.errors.TransactionPushError()
        )):
            return ret_fail

        case _:
            return o.unwrap()


def convert_reward_to_int(reward_str):
    int_part, decimal_part = (
        reward_str.split('.')[0],
        reward_str.split('.')[1].split(' ')[0]
    )
    return int(int_part + decimal_part)


class ContractState:

    def __init__(
        self,
        contract: GPUContractAPI
    ):
        self.contract = contract

        self._config = load_skynet_toml().dgpu
        self._poll_index = 0

        self._queue: list[Request] = []
        self._status_by_rid: dict[int, list[WorkerStatusV0]] = {}
        self._results: list[ResultV0] = []

        self._new_data = trio.Event()

    @property
    def poll_index(self) -> int:
        return self._poll_index

    async def _fetch_results(self):
        self._results = await self.contract.get_worker_results(self._config.account)

    async def _fetch_statuses_for_id(self, rid: int):
        self._status_by_rid[rid] = await self.contract.get_statuses_for_request(rid)

    async def update_state(self):
        '''
        Get a "snapshot" of current contract table state

        '''
        # raw queue from chain
        _queue = await self.contract.get_requests_since(3600)

        # filter out invalids
        self._queue = []
        for req in _queue:
            try:
                msgspec.json.decode(req.body, type=BodyV0)
                self._queue.append(req)

            except msgspec.ValidationError:
                logging.exception(f'dropping req {req.id} due to:')
                ...

        random.shuffle(self._queue)
        self._queue = sorted(
            self._queue,
            key=lambda req: convert_reward_to_int(req.reward),
            reverse=True
        )

        async with trio.open_nursery() as n:
            n.start_soon(self._fetch_results)
            for req in self._queue:
                n.start_soon(
                    self._fetch_statuses_for_id, req.id)


        maybe_update_tui(lambda tui: tui.network_update(self))

    async def wait_data_update(self):
        await self._new_data.wait()

    async def _state_update_task(self, poll_time: float):
        '''
        Long running task, polls gpu contract tables latest table rows,
        awakes any self._data_event waiters

        '''
        while True:
            start_time = time.time()
            await self.update_state()
            elapsed = time.time() - start_time
            self._new_data.set()
            await trio.sleep(max(poll_time - elapsed, 0.1))
            self._new_data = trio.Event()
            self._poll_index += 1

    # views into data

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    @property
    def first(self) -> Request | None:
        if len(self._queue) > 0:
            return self._queue[0]

        else:
            return None

    def competitors_for_id(self, request_id: int) -> set[str]:
        return set((
            status.worker
            for status in self._status_by_rid[request_id]
            if status.worker != self._config.account
        ))

    # predicates

    def is_request_filled(self, request_id: int) -> bool:
        return request_id in [
            result.request_id for result in self._results
        ]

    def is_request_in_progress(self, request_id: int) -> bool:
        return request_id in self._status_by_rid

    def should_compete_for_id(self, request_id: int) -> bool:
        return not bool(
            self._conn.config.non_compete &
            self.competitors_for_id(request_id)
        )

    async def should_cancel_work(self, request_id: int) -> bool:
        logging.info('should cancel work?')
        if request_id not in self._status_by_rid:
            logging.info(f'request #{request_id} no longer in queue, likely its been filled by another worker, cancelling work...')
            return True

        should_cancel = not self.should_compete_for_id(request_id)
        logging.info(f'cancel: {should_cancel}')
        return should_cancel



__state_mngr = None

@acm
async def maybe_open_contract_state_mngr(contract: GPUContractAPI):
    global __state_mngr

    if __state_mngr:
        yield __state_mngr
        return

    config = load_skynet_toml().dgpu

    mngr = ContractState(contract)
    async with trio.open_nursery() as n:
        await mngr.update_state()
        n.start_soon(mngr._state_update_task, config.poll_time)
        __state_mngr = mngr
        yield mngr
        n.cancel_scope.cancel()
