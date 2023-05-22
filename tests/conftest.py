#!/usr/bin/python

import logging

from pathlib import Path

import pytest

from skynet.db import open_new_database
from skynet.nodeos import open_nodeos


@pytest.fixture(scope='session')
def postgres_db():
    with open_new_database() as db_params:
        yield db_params

@pytest.fixture(scope='session')
def cleos():
    with open_nodeos() as cli:
        contract_acc = cli.new_account('telos.gpu', ram=300000)

        cli.new_account(name='testworker1')
        cli.new_account(name='testworker2')
        cli.new_account(name='testworker3')

        cli.deploy_contract_from_host(
            'telos.gpu',
            'tests/contracts/telos.gpu',
            verify_hash=False,
            create_account=False
        )
        yield cli
