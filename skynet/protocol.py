from msgspec import Struct

from skynet.utils import hash_dict


class ModelParams(Struct):
    name: str
    pipe_fqn: str
    setup: dict

    def get_uid(self) -> str:
        return f'{self.pipe_fqn}:{self.name}-{hash_dict(self.setup)}'


class ParamsStruct(Struct):
    model: ModelParams
    runtime_args: list
    runtime_kwargs: dict


class ComputeRequest(Struct):
    method: str
    params: ParamsStruct


# telos.gpu smart contract types

TimestampSec = int


class ConfigRow(Struct):
    token_contract: str
    token_symbol: str
    nonce: int


class AccountRow(Struct):
    user: str
    balance: str


class CardStruct(Struct):
    card_name: str
    version: str
    total_memory: int
    mp_count: int
    extra: str
    is_online: bool


class WorkerRow(Struct):
    account: str
    joined: TimestampSec
    left: TimestampSec
    url: str
    cards: list[CardStruct]


class WorkerStatusStruct(Struct):
    worker: str
    status: str
    started: TimestampSec


class RequestRow(Struct):
    nonce: int
    user: str
    reward: str
    min_verification: int
    body: str
    inputs: list[str]
    status: list[WorkerStatusStruct]
    timestamp: TimestampSec


class WorkerResultRow(Struct):
    id: int
    request_id: int
    user: str
    worker: str
    result_hash: str
    ipfs_hash: str
    submited: TimestampSec
