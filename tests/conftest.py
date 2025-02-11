import pytest

from skynet.ipfs import AsyncIPFSHTTP


@pytest.fixture(scope='session')
def ipfs_client():
    yield AsyncIPFSHTTP('http://127.0.0.1:5001')


@pytest.fixture(scope='session')
def postgres_db():
    from skynet.db import open_new_database
    with open_new_database() as db_params:
        yield db_params


@pytest.fixture(scope='module')
def skynet_cleos(cleos_bs):
    cleos = cleos_bs

    # priv, pub = cleos.create_key_pair()
    # cleos.import_key('gpu.scd', priv)
    cleos.new_account('gpu.scd', ram=4200000)

    cleos.deploy_contract_from_path(
        'gpu.scd',
        'tests/contracts/gpu.scd',
        create_account=False
    )

    cleos.push_action(
        'gpu.scd',
        'config',
        ['eosio.token', '4,TLOS'],
        'gpu.scd'
    )

    cleos.new_account('testworker')

    yield cleos


@pytest.fixture
def inject_mockers():
    from skynet.constants import MODELS
    from skynet.types import ModelDesc

    MODELS['skygpu/txt2img-mocker'] = ModelDesc(
        short='tester',
        mem=0.01,
        attrs={},
        tags=['txt2img']
    )

    yield


@pytest.fixture(scope='session')
def ipfs_node(dockerctl):
    rpc_port = 15001
    with dockerctl.run(
        'ipfs/go-ipfs:latest',
        name='skynet-ipfs',
        ports={
            '8080/tcp': 18080,
            '4001/tcp': 14001,
            '5001/tcp': ('127.0.0.1', rpc_port)
        }
    ) as cntr:
        yield cntr, AsyncIPFSHTTP(f'http://127.0.0.1:{rpc_port}')
