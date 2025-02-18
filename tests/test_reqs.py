import pytest

from skynet.dgpu.compute import maybe_load_model, compute_one


@pytest.mark.parametrize("mode", [
    ('diffuse'), ('txt2img')
])
async def test_pipeline_mocker(inject_mockers, mode):
    model = 'skygpu/txt2img-mocker'
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


async def test_pipeline():
    model = 'stabilityai/stable-diffusion-xl-base-1.0'
    mode = 'txt2img'
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
