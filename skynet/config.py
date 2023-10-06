#!/usr/bin/python

import os
import toml

from pathlib import Path

from .constants import DEFAULT_CONFIG_PATH


class ConfigParsingError(BaseException):
    ...


def load_skynet_toml(file_path=DEFAULT_CONFIG_PATH) -> dict:
    config = toml.load(file_path)
    return config


def load_key(config: dict, key: str) -> str:
    for skey in key.split('.'):
        if skey not in config:
            conf_keys = [k for k in config]
            raise ConfigParsingError(f'key \"{skey}\" not in {conf_keys}')

        config = config[skey]

    return config


def set_hf_vars(hf_token: str, hf_home: str):
    os.environ['HF_TOKEN'] = hf_token
    os.environ['HF_HOME'] = hf_home
    os.environ['HUGGINGFACE_HUB_CACHE'] = hf_home
