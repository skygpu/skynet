#!/usr/bin/python

import time
import random
import string
import logging

from functools import partial

import trio
import pytest
import psycopg2
import trio_asyncio

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from skynet_bot.constants import *
from skynet_bot.brain import run_skynet


@pytest.fixture(scope='session')
def postgres_db(dockerctl):
    rpassword = ''.join(
        random.choice(string.ascii_lowercase)
        for i in range(12))
    password = ''.join(
        random.choice(string.ascii_lowercase)
        for i in range(12))

    with dockerctl.run(
        'postgres',
        command='postgres',
        ports={'5432/tcp': None},
        environment={
            'POSTGRES_PASSWORD': rpassword
        }
    ) as containers:
        container = containers[0]
        # ip = container.attrs['NetworkSettings']['IPAddress']
        port = container.ports['5432/tcp'][0]['HostPort']
        host = f'localhost:{port}'

        for log in container.logs(stream=True):
            log = log.decode().rstrip()
            logging.info(log)
            if ('database system is ready to accept connections' in log or
                'database system is shut down' in log):
                break

        # why print the system is ready to accept connections when its not
        # postgres? wtf
        time.sleep(1)
        logging.info('creating skynet db...')

        conn = psycopg2.connect(
            user='postgres',
            password=rpassword,
            host='localhost',
            port=port
        )
        logging.info('connected...')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(
                f'CREATE USER {DB_USER} WITH PASSWORD \'{password}\'')
            cursor.execute(
                f'CREATE DATABASE {DB_NAME}')
            cursor.execute(
                f'GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER}')

        logging.info('done.')
        yield container, password, host


@pytest.fixture
async def skynet_running(postgres_db):
    db_container, db_pass, db_host = postgres_db
    async with (
        trio_asyncio.open_loop(),
        trio.open_nursery() as n
    ):
        await n.start(
            partial(run_skynet,
                db_pass=db_pass,
                db_host=db_host))

        yield
        n.cancel_scope.cancel()


