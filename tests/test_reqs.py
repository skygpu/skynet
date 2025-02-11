import pytest

from skynet.types import ModelMode, BodyV0Params
from skynet.dgpu.compute import maybe_load_model, compute_one

from skynet._testing import override_dgpu_config


@pytest.mark.parametrize("mode", [
    (ModelMode.DIFFUSE), (ModelMode.TXT2IMG)
])
async def test_pipeline_mocker(inject_mockers, mode):
    override_dgpu_config(
        account='testworker1',
        permission='active',
        key='',
        node_url='',
        ipfs_url='http://127.0.0.1:5001',
        hf_token=''
    )
    model = 'skygpu/txt2img-mocker'
    params = BodyV0Params(
        prompt="Kronos God Realistic 4k",
        model=model,
        step=4,
        width=1024,
        height=1024,
        seed=168402949,
        guidance="7.5"
    )

    with maybe_load_model(model, mode) as model:
        compute_one(model, 0, mode, params)


async def test_pipeline():
    model = 'stabilityai/stable-diffusion-xl-base-1.0'
    mode = 'txt2img'
    params = BodyV0Params(
        prompt="Kronos God Realistic 4k",
        model=model,
        step=21,
        width=1024,
        height=1024,
        seed=168402949,
        guidance="7.5"
    )

    with maybe_load_model(model, mode) as model:
        compute_one(model, 0, mode, params)
