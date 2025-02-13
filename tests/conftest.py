import pytest

from skynet.ipfs import AsyncIPFSHTTP
from skynet.contract import GPUContractAPI
from skynet._testing import override_dgpu_config


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

    contract_path = 'tests/contracts/skygpu-contract/target'
    cleos.deploy_contract_from_path(
        'gpu.scd',
        contract_path,
        contract_name='skygpu',
        create_account=False
    )

    cleos.push_action(
        'gpu.scd',
        'config',
        ['eosio.token', '4,TLOS'],
        'gpu.scd'
    )

    testworker_key = '5KRPFxF4RJebqPXqRzwStmCaEWeRfp3pR7XUNoA3zCHt5fnPu3s'
    pub_key = cleos.import_key('testworker', testworker_key)
    cleos.new_account('testworker', key=pub_key)

    cleos.wait_blocks(1)

    yield GPUContractAPI(cleos), cleos


@pytest.fixture
def inject_mockers():
    from skynet.constants import MODELS
    from skynet.types import ModelDesc, ModelMode

    MODELS['skygpu/mocker'] = ModelDesc(
        short='tester',
        mem=0.01,
        attrs={},
        tags=[
            ModelMode.TXT2IMG,
            ModelMode.IMG2IMG,
            ModelMode.INPAINT
        ]
    )

    MODELS['skygpu/mocker-upscale'] = ModelDesc(
        short='tester-upscale',
        mem=0.01,
        attrs={},
        tags=[
            ModelMode.UPSCALE
        ]
    )

    override_dgpu_config(
        account='testworker1',
        permission='active',
        key='',
        node_url='',
        ipfs_url='http://127.0.0.1:5001',
        hf_token=''
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
        },
        timeout=60
    ) as cntr:
        yield cntr, AsyncIPFSHTTP(f'http://127.0.0.1:{rpc_port}')
