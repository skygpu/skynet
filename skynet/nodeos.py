#!/usr/bin/env python3

import json
import time
import logging

from contextlib import contextmanager as cm

import docker

from leap.cleos import CLEOS
from leap.sugar import get_container, Symbol


@cm
def open_nodeos(cleanup: bool = True):
    dclient = docker.from_env()
    vtestnet = get_container(
        dclient,
        'guilledk/skynet:leap-4.0.1',
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

        priv, pub = cleos.create_key_pair()
        logging.info(f'SUDO KEYS: {(priv, pub)}')

        cleos.setup_wallet(priv)

        genesis = json.dumps({
            "initial_timestamp": '2017-08-29T02:14:00.000',
            "initial_key": pub,
            "initial_configuration": {
                "max_block_net_usage": 1048576,
                "target_block_net_usage_pct": 1000,
                "max_transaction_net_usage": 1048575,
                "base_per_transaction_net_usage": 12,
                "net_usage_leeway": 500,
                "context_free_discount_net_usage_num": 20,
                "context_free_discount_net_usage_den": 100,
                "max_block_cpu_usage": 200000,
                "target_block_cpu_usage_pct": 1000,
                "max_transaction_cpu_usage": 150000,
                "min_transaction_cpu_usage": 100,
                "max_transaction_lifetime": 3600,
                "deferred_trx_expiration_window": 600,
                "max_transaction_delay": 3888000,
                "max_inline_action_size": 4096,
                "max_inline_action_depth": 4,
                "max_authority_depth": 6
            }
        }, indent=4)

        ec, out = cleos.run(
            ['bash', '-c', f'echo \'{genesis}\' > /root/skynet.json'])
        assert ec == 0

        place_holder = 'EOS5fLreY5Zq5owBhmNJTgQaLqQ4ufzXSTpStQakEyfxNFuUEgNs1=KEY:5JnvSc6pewpHHuUHwvbJopsew6AKwiGnexwDRc2Pj2tbdw6iML9'
        sig_provider = f'{pub}=KEY:{priv}'
        nodeos_config_ini = '/root/nodeos/config.ini'
        ec, out = cleos.run(
            ['bash', '-c', f'sed -i -e \'s/{place_holder}/{sig_provider}/g\' {nodeos_config_ini}'])
        assert ec == 0

        cleos.start_nodeos_from_config(
            nodeos_config_ini,
            data_dir='/root/nodeos/data',
            genesis='/root/skynet.json',
            state_plugin=True)

        time.sleep(0.5)
        cleos.wait_blocks(1)
        cleos.boot_sequence(token_sym=Symbol('GPU', 4))

        priv, pub = cleos.create_key_pair()
        cleos.import_key(priv)
        cleos.private_keys['telos.gpu'] = priv
        logging.info(f'GPU KEYS: {(priv, pub)}')
        cleos.new_account('telos.gpu', ram=4200000, key=pub)

        for i in range(1, 4):
            priv, pub = cleos.create_key_pair()
            cleos.import_key(priv)
            cleos.private_keys[f'testworker{i}'] = priv
            logging.info(f'testworker{i} KEYS: {(priv, pub)}')
            cleos.create_account_staked(
                'eosio', f'testworker{i}', key=pub)

        priv, pub = cleos.create_key_pair()
        cleos.import_key(priv)
        logging.info(f'TELEGRAM KEYS: {(priv, pub)}')
        cleos.create_account_staked(
            'eosio', 'telegram', ram=500000, key=pub)

        cleos.transfer_token(
            'eosio', 'telegram', '1000000.0000 GPU', 'Initial testing funds')

        cleos.deploy_contract_from_host(
            'telos.gpu',
            'tests/contracts/telos.gpu',
            verify_hash=False,
            create_account=False
        )

        ec, out = cleos.push_action(
            'telos.gpu',
            'config',
            ['eosio.token', '4,GPU'],
            f'telos.gpu@active'
        )
        assert ec == 0

        ec, out = cleos.transfer_token(
            'telegram', 'telos.gpu', '1000000.0000 GPU', 'Initial testing funds')
        assert ec == 0

        user_row = cleos.get_table(
            'telos.gpu',
            'telos.gpu',
            'users',
            index_position=1,
            key_type='name',
            lower_bound='telegram',
            upper_bound='telegram'
        )
        assert len(user_row) == 1

        yield cleos

    finally:
        # ec, out = cleos.list_all_keys()
        # logging.info(out)
        if cleanup:
            vtestnet.stop()
            vtestnet.remove()
