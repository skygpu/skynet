#!/usr/bin/env python3

import time
import logging

from contextlib import contextmanager as cm

import docker

from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, Symbol, random_string


@cm
def open_cleos(
    node_url: str,
    key: str | None
):
    vtestnet = None
    try:
        dclient = docker.from_env()
        vtestnet = get_container(
            dclient,
            default_nodeos_image(),
            name=f'skynet-wallet-{random_string(size=8)}',
            force_unique=True,
            detach=True,
            network='host',
            remove=True)

        cleos = CLEOS(dclient, vtestnet, url=node_url, remote=node_url)

        if key:
            cleos.setup_wallet(key)

        yield cleos

    finally:
        if vtestnet:
            vtestnet.stop()



@cm
def open_nodeos(cleanup: bool = True):
    dclient = docker.from_env()
    vtestnet = get_container(
        dclient,
        'guilledk/py-eosio:leap-skynet-4.0.0',
        name='skynet-nodeos',
        force_unique=True,
        detach=True,
        network='host')

    try:
        cleos = CLEOS(
            dclient, vtestnet,
            url='http://127.0.0.1:42000',
            remote='http://127.0.0.1:42000'
        )

        cleos.start_keosd()

        cleos.start_nodeos_from_config(
            '/root/nodeos/config.ini',
            data_dir='/root/nodeos/data',
            genesis='/root/nodeos/genesis/skynet.json',
            state_plugin=True)

        time.sleep(0.5)

        public_dev_key = 'EOS5fLreY5Zq5owBhmNJTgQaLqQ4ufzXSTpStQakEyfxNFuUEgNs1'
        cleos.setup_wallet('5JnvSc6pewpHHuUHwvbJopsew6AKwiGnexwDRc2Pj2tbdw6iML9')
        cleos.wait_blocks(1)
        cleos.boot_sequence(token_sym=Symbol('GPU', 4))

        cleos.new_account('telos.gpu', ram=300000, key=public_dev_key)

        for i in range(1, 4):
            cleos.create_account_staked(
                'eosio', f'testworker{i}', key=public_dev_key)

        cleos.create_account_staked(
            'eosio', 'telegram1', ram=500000, key=public_dev_key)

        cleos.deploy_contract_from_host(
            'telos.gpu',
            'tests/contracts/telos.gpu',
            verify_hash=False,
            create_account=False
        )

        yield cleos

    finally:
        # ec, out = cleos.list_all_keys()
        # logging.info(out)
        if cleanup:
            vtestnet.stop()
            vtestnet.remove()
