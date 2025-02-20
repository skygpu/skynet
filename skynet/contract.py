import time

import msgspec
from leap import CLEOS
from leap.protocol import Name

from skynet.types import (
    Config, ConfigV0, ConfigV1,
    Account, AccountV0, AccountV1,
    WorkerV0,
    Request, RequestV0, RequestV1,
    BodyV0,
    WorkerStatusV0,
    ResultV0
)


class ConfigNotFound(BaseException):
    ...

class AccountNotFound(BaseException):
    ...

class WorkerNotFound(BaseException):
    ...

class RequestNotFound(BaseException):
    ...

class WorkerStatusNotFound(BaseException):
    ...


class GPUContractAPI:

    def __init__(self, cleos: CLEOS, proto_version: int = 0):
        self.receiver = 'gpu.scd'
        self._cleos = cleos
        self.proto_version = proto_version

    # views into data

    async def get_config(self) -> Config:
        rows = await self._cleos.aget_table(
            self.receiver, self.receiver, 'config',
            resp_cls=ConfigV1 if self.proto_version > 1 else ConfigV0
        )
        if len(rows) == 0:
            raise ConfigNotFound()

        return rows[0]

    async def get_user(self, user: str) -> Account:
        rows = await self._cleos.aget_table(
            self.receiver, self.receiver, 'users',
            key_type='name',
            lower_bound=user,
            upper_bound=user,
            resp_cls=AccountV1 if self.proto_version > 1 else AccountV0
        )
        if len(rows) == 0:
            raise AccountNotFound(user)

        return rows[0]

    async def get_users(self) -> list[Account]:
        return await self._cleos.aget_table(
            self.receiver, self.receiver, 'users', resp_cls=AccountV1 if self.proto_version > 0 else AccountV0)

    async def get_worker(self, worker: str) -> WorkerV0:
        rows = await self._cleos.aget_table(
            self.receiver, self.receiver, 'workers',
            key_type='name',
            lower_bound=worker,
            upper_bound=worker,
            resp_cls=WorkerV0
        )
        if len(rows) == 0:
            raise WorkerNotFound(worker)

        return rows[0]

    async def get_workers(self) -> list[WorkerV0]:
        return await self._cleos.aget_table(self.receiver, self.receiver, 'workers', resp_cls=WorkerV0)

    async def get_queue(self) -> Request:
        return await self._cleos.aget_table(
            self.receiver, self.receiver, 'queue', resp_cls=RequestV1 if self.proto_version > 0 else RequestV0)

    async def get_request(self, request_id: int) -> Request:
        rows = await self._cleos.aget_table(
            self.receiver, self.receiver, 'queue',
            lower_bound=request_id,
            upper_bound=request_id,
            resp_cls=RequestV1 if self.proto_version > 0  else RequestV0
        )
        if len(rows) == 0:
            raise RequestNotFound(request_id)

        return rows[0]

    async def get_requests_since(self, seconds: int) -> list[Request]:
        return await self._cleos.aget_table(
            self.receiver, self.receiver, 'queue',
            index_position=2,
            key_type='i64',
            lower_bound=int(time.time()) - seconds,
            resp_cls=RequestV1 if self.proto_version > 0  else RequestV0
        )

    async def get_statuses_for_request(self, request_id: int) -> list[WorkerStatusV0]:
        return await self._cleos.aget_table(
            self.receiver, str(Name.from_int(request_id)), 'status',
            resp_cls=WorkerStatusV0
        )

    async def get_worker_status_for_request(self, request_id: int, worker: str) -> WorkerStatusV0:
        rows = await self._cleos.aget_table(
            self.receiver, str(Name.from_int(request_id)), 'status',
            key_type='name',
            lower_bound=worker,
            upper_bound=worker,
            resp_cls=WorkerStatusV0
        )
        if len(rows) == 0:
            raise WorkerStatusNotFound(request_id)

        return rows[0]

    async def get_results(self, request_id: int) -> list[ResultV0]:
        return await self._cleos.aget_table(
            self.receiver, self.receiver, 'results',
            index_position=2,
            key_type='i64',
            lower_bound=request_id,
            upper_bound=request_id,
            resp_cls=ResultV0
        )

    async def get_worker_results(self, worker: str) -> list[ResultV0]:
        return await self._cleos.aget_table(
            self.receiver, self.receiver, 'results',
            index_position=4,
            key_type='name',
            lower_bound=worker,
            upper_bound=worker,
            resp_cls=ResultV0
        )

    # system actions
    async def init_config(self, token_account: str, token_symbol: str):
        return await self._cleos.a_push_action(
            self.receiver,
            'config',
            [token_account, token_symbol],
            self.receiver
        )

    async def clean_tables(self, nuke: bool = False):
        return await self._cleos.a_push_action(
            self.receiver,
            'clean',
            [nuke],
            self.receiver
        )

    # balance actions

    async def deposit(self, user: str, quantity: str):
        return await self._cleos.a_push_action(
            'eosio.token',
            'transfer',
            [user, self.receiver, quantity, 'testing gpu deposit'],
            user,
            key=self._cleos.private_keys[user]
        )

    async def withdraw(self, user: str, quantity: str):
        return await self._cleos.a_push_action(
            self.receiver,
            'withdraw',
            [user, quantity],
            user,
            key=self._cleos.private_keys[user]
        )

    # worker actions

    async def register_worker(
        self,
        worker: str,
        url: str
    ):
        return await self._cleos.a_push_action(
            self.receiver,
            'regworker',
            [worker, url],
            worker,
            key=self._cleos.private_keys[worker]
        )

    async def unregister_worker(
        self,
        worker: str,
        reason: str
    ):
        return await self._cleos.a_push_action(
            self.receiver,
            'unregworker',
            [worker, reason],
            worker,
            key=self._cleos.private_keys[worker]
        )

    async def accept_work(
        self,
        worker: str,
        request_id: int,
        max_workers: int = 10
    ):
        return await self._cleos.a_push_action(
            self.receiver,
            'workbegin',
            [worker, request_id, max_workers],
            worker,
            key=self._cleos.private_keys[worker]
        )

    async def cancel_work(
        self,
        worker: str,
        request_id: int,
        reason: str
    ):
        return await self._cleos.a_push_action(
            self.receiver,
            'workcancel',
            [worker, request_id, reason],
            worker,
            key=self._cleos.private_keys[worker]
        )

    async def submit_work(
        self,
        worker: str,
        request_id: int,
        result_hash: str,
        ipfs_hash: str,
        request_hash: str | None = None
    ):
        args = [worker, request_id, result_hash, ipfs_hash]
        if request_hash:
            args.insert(2, request_hash)

        return await self._cleos.a_push_action(
            self.receiver,
            'submit',
            args,
            worker,
            key=self._cleos.private_keys[worker]
        )

    # user actions

    async def enqueue(
        self,
        account: str,
        body: BodyV0,
        binary_data: str = '',
        reward: str = '1.0000 TLOS',
        min_verification: int = 1
    ) -> int:

        body = msgspec.json.encode(body).decode('utf-8')
        result = await self._cleos.a_push_action(
            self.receiver,
            'enqueue',
            [
                account,
                body,
                binary_data,
                reward,
                min_verification
            ],
            account,
            key=self._cleos.private_keys[account]
        )
        console = result['processed']['action_traces'][0]['console']
        nonce_index = -1
        timestamp_index = -2
        lines = console.rstrip().split('\n')
        nonce = int(lines[nonce_index])
        timestamp = lines[timestamp_index]

        return RequestV1(
            id=int(nonce),
            user=account,
            reward=reward,
            min_verification=min_verification,
            body=body,
            binary_data=binary_data,
            timestamp=timestamp
        )

    async def dequeue(self, user: str, request_id: int):
        return await self._cleos.a_push_action(
            self.receiver,
            'dequeue',
            [user, request_id],
            user,
            key=self._cleos.private_keys[user]
        )
