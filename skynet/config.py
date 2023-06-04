#!/usr/bin/python

import os
import json

from pathlib import Path
from configparser import ConfigParser
from re import sub

from .constants import DEFAULT_CONFIG_PATH


def load_skynet_ini(
    file_path=DEFAULT_CONFIG_PATH
):
    config = ConfigParser()
    config.read(file_path)

    return config


def init_env_from_config(
    hf_token: str | None = None,
    hf_home: str | None = None,
    tg_token: str | None = None,
    file_path=DEFAULT_CONFIG_PATH
):
    config = load_skynet_ini(file_path=file_path)

    if 'HF_TOKEN' in os.environ:
        hf_token = os.environ['HF_TOKEN']

    elif 'skynet.dgpu' in config:
        sub_config = config['skynet.dgpu']
        if 'hf_token' in sub_config:
            hf_token = sub_config['hf_token']

    if 'HF_HOME' in os.environ:
        hf_home = os.environ['HF_HOME']

    elif 'skynet.dgpu' in config:
        sub_config = config['skynet.dgpu']
        if 'hf_home' in sub_config:
            hf_home = sub_config['hf_home']

    if 'TG_TOKEN' in os.environ:
        tg_token = os.environ['TG_TOKEN']
    elif 'skynet.telegram' in config:
        sub_config = config['skynet.telegram']
        if 'token' in sub_config:
            tg_token = sub_config['token']

    return hf_home, hf_token, tg_token


def load_account_info(
    _type: str,
    key: str | None = None,
    account: str | None = None,
    permission: str | None = None,
    file_path=DEFAULT_CONFIG_PATH
):
    config = load_skynet_ini(file_path=file_path)

    type_key = f'skynet.{_type}'

    if type_key in config:
        sub_config = config[type_key]
        if not key and 'key' in sub_config:
            key = sub_config['key']

        if not account and 'name' in sub_config:
            account = sub_config['name']

        if not permission and 'permission' in sub_config:
            permission = sub_config['permission']

    return key, account, permission


def load_endpoint_info(
    _type: str,
    node_url: str | None = None,
    hyperion_url: str | None = None,
    ipfs_url: str | None = None,
    file_path=DEFAULT_CONFIG_PATH
):
    config = load_skynet_ini(file_path=file_path)

    type_key = f'skynet.{_type}'

    if type_key in config:
        sub_config = config[type_key]
        if not node_url and 'node_url' in sub_config:
            node_url = sub_config['node_url']

        if not hyperion_url and 'hyperion_url' in sub_config:
            hyperion_url = sub_config['hyperion_url']

        if not ipfs_url and 'ipfs_url' in sub_config:
            ipfs_url = sub_config['ipfs_url']

    return node_url, hyperion_url, ipfs_url
