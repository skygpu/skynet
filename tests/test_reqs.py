import pytest
from PIL import Image

from skynet.types import ModelMode, BodyV0Params
from skynet.dgpu.compute import maybe_load_model, compute_one


@pytest.mark.parametrize('mode,model', [
    (ModelMode.DIFFUSE, 'skygpu/mocker'),
    (ModelMode.TXT2IMG, 'skygpu/mocker'),
    (ModelMode.IMG2IMG, 'skygpu/mocker'),
    (ModelMode.INPAINT, 'skygpu/mocker'),
    (ModelMode.UPSCALE, 'skygpu/mocker-upscale'),
])
async def test_pipeline_mocker(inject_mockers, mode, model):
    # always insert at least two inputs to make all modes pass
    inputs = [
        Image.new('RGB', (1, 1), color='green')
        for i in range(2)
    ]
    params = BodyV0Params(
        prompt="Kronos God Realistic 4k",
        model=model,
        step=4,
        width=1024,
        height=1024,
        seed=168402949,
        guidance="7.5",
        strength="0.65"
    )

    with maybe_load_model(model, mode) as model:
        compute_one(model, 0, mode, params, inputs)

# disable for now (cuda)
# async def test_pipeline():
#     model = 'stabilityai/stable-diffusion-xl-base-1.0'
#     mode = 'txt2img'
#     params = BodyV0Params(
#         prompt="Kronos God Realistic 4k",
#         model=model,
#         step=21,
#         width=1024,
#         height=1024,
#         seed=168402949,
#         guidance="7.5"
#     )
# 
#     with maybe_load_model(model, mode) as model:
#         compute_one(model, 0, mode, params)
