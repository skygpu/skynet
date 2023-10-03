#!/usr/bin/python

from configparser import ConfigParser

from .constants import DEFAULT_CONFIG_PATH


class ConfigParsingError(BaseException):
    ...


def load_skynet_ini(
    file_path=DEFAULT_CONFIG_PATH
) -> ConfigParser:
    config = ConfigParser()
    config.read(file_path)

    return config


def load_key(config: ConfigParser, section: str, key: str) -> str:
    if section not in config:
        conf_sections = [s for s in config]
        raise ConfigParsingError(f'section \"{section}\" not in {conf_sections}')

    if key not in config[section]:
        conf_keys = [k for k in config[section]]
        raise ConfigParsingError(f'key \"{key}\" not in {conf_keys}')

    return str(config[section][key])
