#!/usr/bin/python

import io
import random
import logging

import torch
import tractor

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler
)

from .types import ImageGenRequest
from .constants import ALGOS


def pipeline_for(algo: str, mem_fraction: float):
    assert torch.cuda.is_available()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(mem_fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    params = {
        'torch_dtype': torch.float16,
        'safety_checker': None
    }

    if algo == 'stable':
        params['revision'] = 'fp16'

    pipe = StableDiffusionPipeline.from_pretrained(
        ALGOS[algo], **params)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)

    return pipe.to("cuda")

@tractor.context
async def open_gpu_worker(
    ctx: tractor.Context,
    start_algo: str,
    mem_fraction: float
):
    current_algo = start_algo
    with torch.no_grad():
        pipe = pipeline_for(current_algo, mem_fraction)
        await ctx.started()

        async with ctx.open_stream() as bus:
            async for ireq in bus:
                if ireq.algo != current_algo:
                    current_algo = ireq.algo
                    pipe = pipeline_for(current_algo, mem_fraction)

                seed = ireq.seed if ireq.seed else random.randint(0, 2 ** 64)
                image = pipe(
                    ireq.prompt,
                    width=ireq.width,
                    height=ireq.height,
                    guidance_scale=ireq.guidance,
                    num_inference_steps=ireq.step,
                    generator=torch.Generator("cuda").manual_seed(seed)
                ).images[0]

                torch.cuda.empty_cache()

                # convert PIL.Image to BytesIO
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                await bus.send(img_bytes.getvalue())

