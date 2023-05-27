#!/usr/bin/env python3

import time
import json

from hashlib import sha256
from functools import partial

import trio
import requests

from skynet.dgpu import open_dgpu_node

from leap.sugar import collect_stdout


def test_enqueue_work(cleos):

    user = cleos.new_account()
    req = json.dumps({
        'method': 'diffuse',
        'params': {
            'algo': 'midj',
            'prompt': 'skynet terminator dystopic',
            'width': 512,
            'height': 512,
            'guidance': 10,
            'step': 28,
            'seed': 420,
            'upscaler': 'x4'
        }
    })
    binary = ''

    ec, out = cleos.push_action(
        'telos.gpu', 'enqueue', [user, req, binary], f'{user}@active'
    )

    assert ec == 0

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 1

    req_on_chain = queue[0]

    assert req_on_chain['user'] == user
    assert req_on_chain['body'] == req
    assert req_on_chain['binary_data'] == binary

    ipfs_hash = None
    sha_hash = None
    for i in range(1, 4):
        trio.run(
            partial(
                open_dgpu_node,
                f'testworker{i}',
                'active',
                cleos,
                initial_algos=['midj']
            )
        )

        if ipfs_hash == None:
            result = cleos.get_table(
                'telos.gpu', 'telos.gpu', 'results',
                index_position=4,
                key_type='name',
                lower_bound=f'testworker{i}',
                upper_bound=f'testworker{i}'
            )
            assert len(result) == 1
            ipfs_hash = result[0]['ipfs_hash']
            sha_hash = result[0]['result_hash']

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 0

    resp = requests.get(f'https://ipfs.io/ipfs/{ipfs_hash}/image.png')
    assert resp.status_code == 200

    assert sha_hash == sha256(resp.content).hexdigest()


def test_enqueue_dequeue(cleos):

    user = cleos.new_account()
    req = json.dumps({
        'method': 'diffuse',
        'params': {
            'algo': 'midj',
            'prompt': 'skynet terminator dystopic',
            'width': 512,
            'height': 512,
            'guidance': 10,
            'step': 28,
            'seed': 420,
            'upscaler': 'x4'
        }
    })
    binary = ''

    ec, out = cleos.push_action(
        'telos.gpu', 'enqueue', [user, req, binary], f'{user}@active'
    )

    assert ec == 0

    request_id = int(collect_stdout(out))

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 1

    ec, out = cleos.push_action(
        'telos.gpu', 'dequeue', [user, request_id], f'{user}@active'
    )

    assert ec == 0

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 0
