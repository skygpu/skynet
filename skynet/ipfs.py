#!/usr/bin/python

import logging

from pathlib import Path
from contextlib import contextmanager as cm

import docker

from docker.models.containers import Container


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

@cm
def open_ipfs_node():
    dclient = docker.from_env()

    container = dclient.containers.run(
        'ipfs/go-ipfs:latest',
        name='skynet-ipfs',
        ports={
            '8080/tcp': 8080,
            '4001/tcp': 4001,
            '5001/tcp': ('127.0.0.1', 5001)
        },
        volumes=[
            str(Path().resolve() / 'tmp/ipfs-docker-staging') + ':/export',
            str(Path().resolve() / 'tmp/ipfs-docker-data') + ':/data/ipfs'
        ],
        detach=True,
        remove=True
    )
    try:

        for log in container.logs(stream=True):
            log = log.decode().rstrip()
            logging.info(log)
            if 'Daemon is ready' in log:
                break

        yield IPFSDocker(container)

    finally:
        if container:
            container.stop()

