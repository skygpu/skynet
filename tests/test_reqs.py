import json

from skynet.constants import *
from skynet.dgpu.compute import maybe_load_model, compute_one


async def test_diffuse(inject_mockers):
    model = 'skygpu/txt2img-mocker'
    mode = 'diffuse'
    params = {
        "prompt": "Kronos God Realistic 4k",
        "model": model,
        "step": 21,
        "width": 1024,
        "height": 1024,
        "seed": 168402949,
        "guidance": "7.5"
    }

    with maybe_load_model(model, mode) as model:
        compute_one(model, 0, mode, params)
