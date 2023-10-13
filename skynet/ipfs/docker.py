#!/usr/bin/python

import sys
import logging

from pathlib import Path
from contextlib import contextmanager as cm

import docker

from docker.types import Mount


@cm
def open_ipfs_node(
    name: str = 'skynet-ipfs',
    teardown: bool = False,
    peers: list[str] = []
):
    dclient = docker.from_env()

    container = None
    try:
        container = dclient.containers.get(name)

    except docker.errors.NotFound:
        data_dir = Path().resolve() / 'ipfs-docker-data'
        data_dir.mkdir(parents=True, exist_ok=True)

        data_target = '/data/ipfs'

        container = dclient.containers.run(
            'ipfs/go-ipfs:latest',
            name='skynet-ipfs',
            ports={
                '8080/tcp': 8080,
                '4001/tcp': 4001,
                '5001/tcp': ('127.0.0.1', 5001)
            },
            mounts=[
                Mount(data_target, str(data_dir), 'bind')
            ],
            detach=True,
            remove=True
        )

        uid, gid = 1000, 1000

        if sys.platform != 'win32':
            ec, out = container.exec_run(['chown', f'{uid}:{gid}', '-R', data_target])
            logging.info(out)
            assert ec == 0

        for log in container.logs(stream=True):
            log = log.decode().rstrip()
            logging.info(log)
            if 'Daemon is ready' in log:
                break

    for peer in peers:
        ec, out = container.exec_run(
            ['ipfs', 'swarm', 'connect', peer])
        if ec != 0:
            logging.error(out)

    yield

    if teardown and container:
        container.stop()
