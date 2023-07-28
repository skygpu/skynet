#!/usr/bin/env python3

import time
import json

from hashlib import sha256
from functools import partial

import trio
import requests
from skynet.constants import DEFAULT_IPFS_REMOTE

from skynet.dgpu import open_dgpu_node

from leap.sugar import collect_stdout


def test_enqueue_work(cleos):
    user = 'telegram'
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
        'telos.gpu', 'enqueue', [user, req, binary, '20.0000 GPU', 1], f'{user}@active'
    )

    assert ec == 0

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 1

    req_on_chain = queue[0]

    assert req_on_chain['user'] == user
    assert req_on_chain['body'] == req
    assert req_on_chain['binary_data'] == binary

    trio.run(
        partial(
            open_dgpu_node,
            f'testworker1',
            'active',
            cleos,
            DEFAULT_IPFS_REMOTE,
            cleos.private_keys['testworker1'],
            initial_algos=['midj']
        )
    )

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 0


def test_enqueue_dequeue(cleos):
    user = 'telegram'
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
        'telos.gpu', 'enqueue', [user, req, binary, '20.0000 GPU', 1], f'{user}@active'
    )

    assert ec == 0

    request_id, _ = collect_stdout(out).split(':')
    request_id = int(request_id)

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 1

    ec, out = cleos.push_action(
        'telos.gpu', 'dequeue', [user, request_id], f'{user}@active'
    )

    assert ec == 0

    queue = cleos.get_table('telos.gpu', 'telos.gpu', 'queue')

    assert len(queue) == 0
