import time

from PIL import Image

import msgspec


__model = {
    'name': 'skygpu/mocker'
}

class MockPipelineResult(msgspec.Struct):
    images: list[Image]

class MockPipeline:

    def __call__(
        self,
        prompt: str,
        *args,
        num_inference_steps: int = 3,
        callback=None,
        mock_step_time: float = 0.1,
        **kwargs
    ):
        for i in range(num_inference_steps):
            time.sleep(mock_step_time)
            if callback:
                callback(i+1)

        img = Image.new('RGB', (1, 1), color='green')

        return MockPipelineResult(images=[img])


def pipeline_for(
    model: str,
    mode: str,
    mem_fraction: float = 1.0,
    cache_dir: str | None = None
):
    return MockPipeline()
