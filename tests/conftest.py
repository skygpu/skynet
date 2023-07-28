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
        yield cli
