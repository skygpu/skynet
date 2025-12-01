from PIL import Image

import msgspec

from skynet.dgpu.utils import convert_from_image_to_cv2


__model = {
    'name': 'skygpu/mocker-upscale'
}

class MockPipelineResult(msgspec.Struct):
    images: list[Image]

class MockUpscalePipeline:

    def enhance(
        self,
        img,
        outscale: int
    ):
        img = Image.new('RGB', (outscale, outscale), color='green')
        output = convert_from_image_to_cv2(img)

        return (output, None)


def pipeline_for(
    model: str,
    mode: str,
    mem_fraction: float = 1.0,
    cache_dir: str | None = None
):
    return MockUpscalePipeline()
