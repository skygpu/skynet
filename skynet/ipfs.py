#!/usr/bin/python

import os
import logging

from pathlib import Path
from contextlib import contextmanager as cm

import asks
import docker

from asks.errors import RequestTimeout
from docker.types import Mount
from docker.models.containers import Container


async def get_ipfs_file(ipfs_link: str):
    logging.info(f'attempting to get image at {ipfs_link}')
    resp = None
    for i in range(10):
        try:
            resp = await asks.get(ipfs_link, timeout=3)

        except asks.errors.RequestTimeout:
            logging.warning('timeout...')

    if resp:
        logging.info(f'status_code: {resp.status_code}')
    else:
        logging.error(f'timeout')
    return resp


class IPFSDocker:

    def __init__(self, container: Container):
        self._container = container

    def add(self, file: str) -> str:
        ec, out = self._container.exec_run(
            ['ipfs', 'add', '-w', f'/export/{file}', '-Q'])
        assert ec == 0

        return out.decode().rstrip()

    def pin(self, ipfs_hash: str):
        ec, out = self._container.exec_run(
            ['ipfs', 'pin', 'add', ipfs_hash])
        assert ec == 0

    def connect(self, remote_node: str):
        ec, out = self._container.exec_run(
            ['ipfs', 'swarm', 'connect', remote_node])
        if ec != 0:
            logging.error(out)

        assert ec == 0


@cm
def open_ipfs_node(name='skynet-ipfs'):
    dclient = docker.from_env()

    try:
        container = dclient.containers.get(name)

    except docker.errors.NotFound:
        staging_dir = Path().resolve() / 'ipfs-docker-staging'
        staging_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path().resolve() / 'ipfs-docker-data'
        data_dir.mkdir(parents=True, exist_ok=True)

        export_target = '/export'
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
                Mount(export_target, str(staging_dir), 'bind'),
                Mount(data_target, str(data_dir), 'bind')
            ],
            detach=True,
            remove=True
        )

        uid = os.getuid()
        gid = os.getgid()
        ec, out = container.exec_run(['chown', f'{uid}:{gid}', '-R', export_target])
        logging.info(out)
        assert ec == 0
        ec, out = container.exec_run(['chown', f'{uid}:{gid}', '-R', data_target])
        logging.info(out)
        assert ec == 0

        for log in container.logs(stream=True):
            log = log.decode().rstrip()
            logging.info(log)
            if 'Daemon is ready' in log:
                break

    yield IPFSDocker(container)

