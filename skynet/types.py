from enum import StrEnum

from msgspec import Struct


class ModelMode(StrEnum):
    DIFFUSE = 'diffuse'
    TXT2IMG = 'txt2img'
    IMG2IMG = 'img2img'
    UPSCALE = 'upscale'
    INPAINT = 'inpaint'


class ModelDesc(Struct):
    short: str  # short unique name
    mem: float  # recomended mem
    attrs: dict  # additional mode specific attrs
    tags: list[ModelMode]

# smart contract types
# https://github.com/guilledk/telos.contracts/blob/gpu_contracts/contracts/telos.gpu/include/telos.gpu/telos.gpu.hpp

'''
ConfigV0

singleton containing global info about system, definition:

```c++
struct [[eosio::table]] global_configuration_struct {
    name token_contract;
    symbol token_symbol;
} global_config_row;

typedef eosio::singleton<"config"_n, global_configuration_struct> global_config;
```
'''

class ConfigV0:
    token_contract: str
    token_symbol: str

'''
RequestV0

a request placed on the queue, definition:

scope: get_self()

```c++
struct [[eosio::table]] work_request_struct {
    uint64_t id;
    name user;
    asset reward;
    uint32_t min_verification;

    uint64_t nonce;

    string body;
    string binary_data;

    time_point_sec timestamp;

    uint64_t primary_key() const { return id; }
    uint64_t by_time() const { return (uint64_t)timestamp.sec_since_epoch(); }
};

typedef eosio::multi_index<
    "queue"_n,
    work_request_struct,
    indexed_by<
        "bytime"_n, const_mem_fun<work_request_struct, uint64_t, &work_request_struct::by_time>
    >
> work_queue;
```
'''

class BodyV0Params(Struct):
    prompt: str
    model: str
    seed: int
    step: int = 1
    guidance: str | float | None = None
    width: int | None = None
    height: int | None = None
    strength: str | float | None = None
    output_type: str | None = 'png'
    upscaler: str | None = None


class BodyV0(Struct):
    method: ModelMode
    params: BodyV0Params


class RequestV0(Struct):
    id: int
    user: str
    reward: str
    min_verification: int
    nonce: int
    body: str
    binary_data: str
    timestamp: str


'''
AccountV0

a user account, users must deposit tokens in order to enqueue requests, definition:

scope: get_self()

```c++
struct [[eosio::table]] account {
    name user;
    asset    balance;
    uint64_t nonce;

    uint64_t primary_key()const { return user.value; }
};
typedef eosio::multi_index<"users"_n, account> users;
```
'''

class AccountV0(Struct):
    user: str
    balance: str
    nonce: int


'''
WorkerV0

a registered worker info, definition:

scope: get_self()

```c++
struct [[eosio::table]] worker {
    name account;
    time_point_sec joined;
    time_point_sec left;
    string url;

    uint64_t primary_key()const { return account.value; }
};
typedef eosio::multi_index<"workers"_n, worker> workers;
```
'''

class WorkerV0(Struct):
    account: str
    joined: str
    left: str
    url: str

'''
WorkerStatusV0

a worker's status related to a currently in progress fill, definition:

scope: request id

```c++
struct [[eosio::table]] worker_status_struct {
    name worker;
    string status;
    time_point_sec started;

    uint64_t primary_key() const { return worker.value; }
};
```
'''

class WorkerStatusV0(Struct):
    worker: str
    status: str
    started: str

'''
ResultV0

a submited result related to a request, definition:

scope: get_self()

```c++
struct [[eosio::table]] work_result_struct {
    uint64_t id;
    uint64_t request_id;
    name user;
    name worker;
    checksum256 result_hash;
    string ipfs_hash;
    time_point_sec submited;

    uint64_t primary_key() const { return id; }
    uint64_t by_request_id() const { return request_id; }
    checksum256 by_result_hash() const { return result_hash; }
    uint64_t by_worker() const { return worker.value; }
    uint64_t by_time() const { return (uint64_t)submited.sec_since_epoch(); }
};

typedef eosio::multi_index<
    "results"_n,
    work_result_struct,
    indexed_by<
        "byreqid"_n, const_mem_fun<work_result_struct, uint64_t, &work_result_struct::by_request_id>
    >,
    indexed_by<
        "byresult"_n, const_mem_fun<work_result_struct, checksum256, &work_result_struct::by_result_hash>
    >,
    indexed_by<
        "byworker"_n, const_mem_fun<work_result_struct, uint64_t, &work_result_struct::by_worker>
    >,
    indexed_by<
        "bytime"_n, const_mem_fun<work_result_struct, uint64_t, &work_result_struct::by_time>
    >
> work_results;
```
'''

class ResultV0(Struct):
    id: int
    request_id: int
    user: str
    worker: str
    result_hash: str
    ipfs_hash: str
    submited: str
