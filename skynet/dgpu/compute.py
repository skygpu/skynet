#!/usr/bin/python

# Skynet Memory Manager

import gc
import logging

from copy import deepcopy
from typing import Any

from diffusers import DiffusionPipeline

import trio
import torch

from skynet.constants import DEFAULT_INITAL_MODELS, MODELS
from skynet.dgpu.errors import DGPUComputeError, DGPUInferenceCancelled
from skynet.protocol import ComputeRequest, ModelParams, ParamsStruct

from skynet.cuda_utils import (
    init_upscaler,
    pipeline_for_v2
)


def unpack_diffuse_params(params: ParamsStruct):
    kwargs = deepcopy(params.runtime_kwargs)

    if 'generator' in kwargs:
        kwargs['generator'] = torch.manual_seed(int(kwargs['generator']))

    return params.runtime_args, kwargs


class SkynetMM:

    def __init__(self, config: dict):
        self.upscaler = init_upscaler()
        self.initial_models: list[ModelParams] = (
            [ModelParams(**model) for model in config['initial_models']]
            if 'initial_models' in config else DEFAULT_INITAL_MODELS
        )

        self.cache_dir = None
        if 'hf_home' in config:
            self.cache_dir = config['hf_home']

        self.num_gpus = torch.cuda.device_count()
        self.gpus = [
            torch.cuda.get_device_properties(i)
            for i in range(self.num_gpus)
        ]

        self._models = {}
        for model in self.initial_models:
            self.load_model(model)

    def log_debug_info(self):
        logging.info('memory summary:')
        logging.info('\n' + torch.cuda.memory_summary())

    def is_model_loaded(self, model: ModelParams):
        return model.get_uid() in self._models

    def load_model(
        self,
        model: ModelParams
    ):
        logging.info(f'loading model {model.name}...')
        if len(self._models.keys()) == 0:
            pipe = pipeline_for_v2(
                model, cache_dir=self.cache_dir)

            self._models[model.get_uid()] = {
                'pipe': pipe,
                'params': model,
                'generated': 0
            }

        else:
            least_used = list(self._models.keys())[0]

            for model in self._models:
                if self._models[
                    least_used]['generated'] > self._models[model]['generated']:
                    least_used = model

            del self._models[least_used]

            logging.info(f'swapping model {least_used} for {model.get_uid()}...')

            gc.collect()
            torch.cuda.empty_cache()

            pipe = pipeline_for_v2(
                model, cache_dir=self.cache_dir)

            self._models[model.get_uid()] = {
                'pipe': pipe,
                'params': model,
                'generated': 0
            }

        logging.info(f'loaded model {model.name}')
        return pipe

    def get_model(self, model: ModelParams) -> DiffusionPipeline:
        if model.name not in MODELS:
            raise DGPUComputeError(f'Unknown model {model.name}')

        if not self.is_model_loaded(model):
            pipe = self.load_model(model)

        else:
            pipe = self._models[model.get_uid()]['pipe']

        return pipe

    def compute_one(
        self,
        request_id: int,
        request: ComputeRequest,
        inputs: list[tuple[Any, str]]
    ) -> list[tuple[bytes, str]]:
        def maybe_cancel_work(step, *args, **kwargs):
            if self._should_cancel:
                should_raise = trio.from_thread.run(self._should_cancel, request_id)
                if should_raise:
                    logging.warn(f'cancelling work at step {step}')
                    raise DGPUInferenceCancelled()

        maybe_cancel_work(0)

        output_type = 'png'
        if 'output_type' in request.params.runtime_kwargs:
            output_type = request.params.runtime_kwargs['output_type']

        outputs = None
        try:
            match request.method:
                case 'diffuse':
                    model = self.get_model(request.params.model)

                    args, kwargs = unpack_diffuse_params(request.params)

                    outputs = model(
                        *args, **kwargs,
                        callback=maybe_cancel_work,
                        callback_steps=1
                    )

                    output = outputs.images[0]

                case _:
                    raise DGPUComputeError('Unsupported compute method')

        except BaseException as e:
            logging.error(e)
            raise DGPUComputeError(str(e))

        finally:
            torch.cuda.empty_cache()

        return [(output, output_type)]
