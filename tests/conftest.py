#!/usr/bin/python

import pytest

from skynet.config import *
from skynet.ipfs import AsyncIPFSHTTP
from skynet.ipfs.docker import open_ipfs_node
from skynet.nodeos import open_nodeos


@pytest.fixture(scope='session')
def ipfs_client():
    with open_ipfs_node(teardown=True):
        yield AsyncIPFSHTTP('http://127.0.0.1:5001')

@pytest.fixture(scope='session')
def postgres_db():
    from skynet.db import open_new_database
    with open_new_database() as db_params:
        yield db_params

@pytest.fixture(scope='session')
def cleos():
    with open_nodeos() as cli:
        yield cli

@pytest.fixture(scope='session')
def dgpu():
    from skynet.dgpu.network import SkynetGPUConnector
    from skynet.dgpu.compute import SkynetMM
    from skynet.dgpu.daemon import SkynetDGPUDaemon

    config = load_skynet_toml(file_path='skynet.toml')
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)
    config = config['skynet']['dgpu']
    conn = SkynetGPUConnector(config)
    mm = SkynetMM(config)
    daemon = SkynetDGPUDaemon(mm, conn, config)

    yield conn, mm, daemon
