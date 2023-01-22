#!/usr/bin/python

import os
import json
import time
import logging

from pathlib import Path
from functools import partial

import pytest

from docker.types import Mount, DeviceRequest

from skynet.db import open_new_database
from skynet.brain import run_skynet
from skynet.network import get_random_port
from skynet.constants import *


@pytest.fixture(scope='session')
def postgres_db(dockerctl):
    with open_new_database() as db_params:
        yield db_params


@pytest.fixture
async def skynet_running():
    async with run_skynet():
        yield


@pytest.fixture
def dgpu_workers(request, dockerctl, skynet_running):
    devices = [DeviceRequest(capabilities=[['gpu']])]
    mounts = [Mount(
        '/skynet', str(Path().resolve()), type='bind')]

    num_containers, initial_algos = request.param

    cmds = []
    for i in range(num_containers):
        dgpu_addr = f'tcp://127.0.0.1:{get_random_port()}'
        cmd = f'''
        pip install -e . && \
        skynet run dgpu \
        --algos=\'{json.dumps(initial_algos)}\' \
        --uid=dgpu-{i} \
        --dgpu={dgpu_addr}
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
        num=num_containers,
    ) as containers:
        yield containers

        #for i, container in enumerate(containers):
        #    logging.info(f'container {i} logs:')
        #    logging.info(container.logs().decode())
