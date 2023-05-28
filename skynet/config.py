#!/usr/bin/python

import os

from pathlib import Path
from configparser import ConfigParser

from .constants import DEFAULT_CONFIG_PATH


def load_skynet_ini(
    file_path=DEFAULT_CONFIG_PATH
):
    config = ConfigParser()
    config.read(file_path)
    return config


def init_env_from_config(
    file_path=DEFAULT_CONFIG_PATH
):
    config = load_skynet_ini()

    if 'HF_TOKEN' in os.environ:
        hf_token = os.environ['HF_TOKEN']
    else:
        hf_token = config['skynet.dgpu']['hf_token']

    if 'HF_HOME' in os.environ:
        hf_home = os.environ['HF_HOME']
    else:
        hf_home = config['skynet.dgpu']['hf_home']

    if 'TG_TOKEN' in os.environ:
        tg_token = os.environ['TG_TOKEN']
    else:
        tg_token = config['skynet.telegram']['token']

    return hf_home, hf_token, tg_token, config


def load_account_info(
    key, account, permission,
    file_path=DEFAULT_CONFIG_PATH
):
    _, _, _, config = init_env_from_config()

    if not key:
        key = config['skynet.account']['key']

    if not account:
        account = config['skynet.account']['name']

    if not permission:
        permission = config['skynet.account']['permission']

    return key, account, permission
