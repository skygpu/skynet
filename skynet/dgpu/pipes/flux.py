import torch

from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    FluxTransformer2DModel
)
from transformers import T5EncoderModel, BitsAndBytesConfig

from huggingface_hub import hf_hub_download

__model = {
    'name': 'black-forest-labs/FLUX.1-schnell'
}

def pipeline_for(
    model: str,
    mode: str,
    mem_fraction: float = 1.0,
    cache_dir: str | None = None
) -> DiffusionPipeline:
    qonfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    params = {
        'torch_dtype': torch.bfloat16,
        'cache_dir': cache_dir,
        'device_map': 'balanced',
        'max_memory': {'cpu': '10GiB', 0: '11GiB'}
        # 'max_memory': {0: '11GiB'}
    }

    text_encoder = T5EncoderModel.from_pretrained(
        'black-forest-labs/FLUX.1-schnell',
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
        quantization_config=qonfig
    )
    params['text_encoder_2'] = text_encoder

    pipe = FluxPipeline.from_pretrained(
        model, **params)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    return pipe
