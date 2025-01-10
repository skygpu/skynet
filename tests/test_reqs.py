
from skynet.config import *

async def test_txt2img():
    req = {
        'id': 0,
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
        'inputs': [],
    }

    config = load_skynet_toml(file_path=config_path)
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)

    assert 'skynet' in config
    assert 'dgpu' in config['skynet']

    mm = SkynetMM(config['skynet']['dgpu'])

    mm.maybe_serve_one(req)


async def test_img2img():
    req = {
        'id': 0,
        'body': json.dumps({
            "method": "img2img",
            "params": {
                "prompt": "Kronos God Realistic 4k",
                "model": list(MODELS.keys())[-2],
                "step": 21,
                "width": 1024,
                "height": 1024,
                "seed": 168402949,
                "guidance": "7.5",
                "strength": "0.5"
            }
        }),
        'inputs': ['QmZcGdXXVQfpco1G3tr2CGFBtv8xVsCwcwuq9gnJBWDymi'],
    }

    config = load_skynet_toml(file_path=config_path)
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)

    assert 'skynet' in config
    assert 'dgpu' in config['skynet']

    mm = SkynetMM(config['skynet']['dgpu'])

    mm.maybe_serve_one(req)

async def test_inpaint():
    req = {
        'id': 0,
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
        'inputs': [
            'QmZcGdXXVQfpco1G3tr2CGFBtv8xVsCwcwuq9gnJBWDymi',
            'Qmccx1aXNmq5mZDS3YviUhgGHXWhQeHvca3AgA7MDjj2hR'
        ],
    }

    config = load_skynet_toml(file_path=config_path)
    hf_token = load_key(config, 'skynet.dgpu.hf_token')
    hf_home = load_key(config, 'skynet.dgpu.hf_home')
    set_hf_vars(hf_token, hf_home)

    assert 'skynet' in config
    assert 'dgpu' in config['skynet']

    mm = SkynetMM(config['skynet']['dgpu'])

    mm.maybe_serve_one(req)
