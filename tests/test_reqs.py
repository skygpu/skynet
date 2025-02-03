import json

from skynet.dgpu.compute import ModelMngr
from skynet.constants import *
from skynet.config import *


async def test_diffuse(dgpu):
    conn, mm, daemon = dgpu
    await conn.cancel_work(0, 'testing')

    daemon._snap['requests'][0] = {}
    req = {
        'id': 0,
        'nonce': 0,
        'body': json.dumps({
            "method": "diffuse",
            "params": {
                "prompt": "Kronos God Realistic 4k",
                "model": list(MODELS.keys())[-1],
                "step": 21,
                "width": 1024,
                "height": 1024,
                "seed": 168402949,
                "guidance": "7.5"
            }
        }),
        'binary_data': '',
    }

    await daemon.maybe_serve_one(req)


async def test_txt2img(dgpu):
    conn, mm, daemon = dgpu
    await conn.cancel_work(0, 'testing')

    daemon._snap['requests'][0] = {}
    req = {
        'id': 0,
        'nonce': 0,
        'body': json.dumps({
            "method": "txt2img",
            "params": {
                "prompt": "Kronos God Realistic 4k",
                "model": list(MODELS.keys())[-1],
                "step": 21,
                "width": 1024,
                "height": 1024,
                "seed": 168402949,
                "guidance": "7.5"
            }
        }),
        'binary_data': '',
    }

    await daemon.maybe_serve_one(req)


async def test_img2img(dgpu):
    conn, mm, daemon = dgpu
    await conn.cancel_work(0, 'testing')

    daemon._snap['requests'][0] = {}
    req = {
        'id': 0,
        'nonce': 0,
        'body': json.dumps({
            "method": "img2img",
            "params": {
                "prompt": "a hindu cat god feline god on a house roof",
                "model": list(MODELS.keys())[-2],
                "step": 21,
                "width": 1024,
                "height": 1024,
                "seed": 168402949,
                "guidance": "7.5",
                "strength": "0.5"
            }
        }),
        'binary_data': 'QmZcGdXXVQfpco1G3tr2CGFBtv8xVsCwcwuq9gnJBWDymi',
    }

    await daemon.maybe_serve_one(req)

async def test_inpaint(dgpu):
    conn, mm, daemon = dgpu
    await conn.cancel_work(0, 'testing')

    daemon._snap['requests'][0] = {}
    req = {
        'id': 0,
        'nonce': 0,
        'body': json.dumps({
            "method": "inpaint",
            "params": {
                "prompt": "a black panther on a sunny roof",
                "model": list(MODELS.keys())[-3],
                "step": 21,
                "width": 1024,
                "height": 1024,
                "seed": 168402949,
                "guidance": "7.5",
                "strength": "0.5"
            }
        }),
        'binary_data':
            'QmZcGdXXVQfpco1G3tr2CGFBtv8xVsCwcwuq9gnJBWDymi,' +
            'Qmccx1aXNmq5mZDS3YviUhgGHXWhQeHvca3AgA7MDjj2hR'
    }

    await daemon.maybe_serve_one(req)
