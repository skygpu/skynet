#!/usr/bin/python

import pytest

from skynet.db import open_new_database
from skynet.ipfs import AsyncIPFSHTTP
from skynet.ipfs.docker import open_ipfs_node
from skynet.nodeos import open_nodeos


@pytest.fixture(scope='session')
def ipfs_client():
    with open_ipfs_node(teardown=True):
        yield AsyncIPFSHTTP('http://127.0.0.1:5001')

@pytest.fixture(scope='session')
def postgres_db():
    with open_new_database() as db_params:
        yield db_params

@pytest.fixture(scope='session')
def cleos():
    with open_nodeos() as cli:
        yield cli
