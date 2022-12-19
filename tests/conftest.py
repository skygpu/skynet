#!/usr/bin/python

import os
import json
import time
import random
import string
import logging

from functools import partial
from pathlib import Path

import trio
import pytest
import psycopg2
import trio_asyncio

from docker.types import Mount, DeviceRequest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from skynet.constants import *
from skynet.brain import run_skynet


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
        name='skynet-test-postgres',
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

        conn.close()

        logging.info('done.')
        yield container, password, host


@pytest.fixture
async def skynet_running(postgres_db):
    db_container, db_pass, db_host = postgres_db

    async with run_skynet(
        db_pass=db_pass,
        db_host=db_host
    ):
        yield


@pytest.fixture
def dgpu_workers(request, dockerctl, skynet_running):
    devices = [DeviceRequest(capabilities=[['gpu']])]
    mounts = [Mount(
        '/skynet', str(Path().resolve()), type='bind')]

    num_containers, initial_algos = request.param

    cmds = []
    for i in range(num_containers):
        cmd = f'''
        pip install -e . && \
        skynet run dgpu \
        --algos=\'{json.dumps(initial_algos)}\' \
        --uid=dgpu-{i}
        '''
        cmds.append(['bash', '-c', cmd])

    logging.info(f'launching: \n{cmd}')

    with dockerctl.run(
        DOCKER_RUNTIME_CUDA,
        name='skynet-test-runtime-cuda',
        commands=cmds,
        environment={
            'HF_TOKEN': os.environ['HF_TOKEN'],
            'HF_HOME': '/skynet/hf_home'
        },
        network='host',
        mounts=mounts,
        device_requests=devices,
        num=num_containers
    ) as containers:
        yield containers

        #for i, container in enumerate(containers):
        #    logging.info(f'container {i} logs:')
        #    logging.info(container.logs().decode())
