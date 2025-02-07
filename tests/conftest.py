import pytest

from skynet.config import *
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

    priv, pub = cleos.create_key_pair()
    cleos.import_key('telos.gpu', priv)
    cleos.new_account('telos.gpu', ram=4200000, key=pub)

    cleos.deploy_contract_from_path(
        'telos.gpu',
        'tests/contracts/telos.gpu',
        create_account=False
    )

    cleos.push_action(
        'telos.gpu',
        'config',
        ['eosio.token', '4,GPU'],
        'telos.gpu'
    )

    yield cleos


@pytest.fixture(scope='session')
def dgpu():
    from skynet.dgpu.network import NetConnector
    from skynet.dgpu.compute import ModelMngr
    from skynet.dgpu.daemon import WorkerDaemon

    config = load_skynet_toml(file_path='skynet.toml')
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)
    config = config['skynet']['dgpu']
    conn = NetConnector(config)
    mm = ModelMngr(config)
    daemon = WorkerDaemon(mm, conn, config)

    yield conn, mm, daemon
