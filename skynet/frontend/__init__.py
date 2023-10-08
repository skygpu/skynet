#!/usr/bin/python

import random

from ..constants import *


class ConfigRequestFormatError(BaseException):
    ...

class ConfigUnknownAttribute(BaseException):
    ...

class ConfigUnknownAlgorithm(BaseException):
    ...

class ConfigUnknownUpscaler(BaseException):
    ...

class ConfigUnknownAutoConfSetting(BaseException):
    ...

class ConfigSizeDivisionByEight(BaseException):
    ...



def validate_user_config_request(req: str):
    params = req.split(' ')

    if len(params) < 3:
        raise ConfigRequestFormatError('config request format incorrect')

    else:
        try:
            attr = params[1]

            match attr:
                case 'model' | 'algo':
                    attr = 'model'
                    val = params[2]
                    shorts = [model_info['short'] for model_info in MODELS.values()]
                    if val not in shorts:
                        raise ConfigUnknownAlgorithm(f'no model named {val}')

                    val = get_model_by_shortname(val)

                case 'step':
                    val = int(params[2])
                    val = max(min(val, MAX_STEP), MIN_STEP)

                case 'width':
                    val = max(min(int(params[2]), MAX_WIDTH), 16)
                    if val % 8 != 0:
                        raise ConfigSizeDivisionByEight(
                            'size must be divisible by 8!')

                case 'height':
                    val = max(min(int(params[2]), MAX_HEIGHT), 16)
                    if val % 8 != 0:
                        raise ConfigSizeDivisionByEight(
                            'size must be divisible by 8!')

                case 'seed':
                    val = params[2]
                    if val == 'auto':
                        val = None
                    else:
                        val = int(params[2])

                case 'guidance':
                    val = float(params[2])
                    val = max(min(val, MAX_GUIDANCE), 0)

                case 'strength':
                    val = float(params[2])
                    val = max(min(val, 0.99), 0.01)

                case 'upscaler':
                    val = params[2]
                    if val == 'off':
                        val = None
                    elif val != 'x4':
                        raise ConfigUnknownUpscaler(
                            f'\"{val}\" is not a valid upscaler')

                case 'autoconf':
                    val = params[2]
                    if val == 'on':
                        val = True

                    elif val == 'off':
                        val = False

                    else:
                        raise ConfigUnknownAutoConfSetting(
                            f'\"{val}\" not a valid setting for autoconf')

                case _:
                    raise ConfigUnknownAttribute(
                        f'\"{attr}\" not a configurable parameter')

            display_val = val
            if attr == 'seed':
                if not val:
                    display_val = 'Random'

            return attr, val, f'config updated! {attr} to {display_val}'

        except ValueError:
            raise ValueError(f'\"{val}\" is not a number silly')


def perform_auto_conf(config: dict) -> dict:
    model = config['model']
    prefered_size_w = 512
    prefered_size_h = 512

    if 'xl' in model:
        prefered_size_w = 1024
        prefered_size_h = 1024

    else:
        prefered_size_w = 512
        prefered_size_h = 512

    config['step'] = random.randint(20, 35)
    config['width'] = prefered_size_w
    config['height'] = prefered_size_h

    return config
