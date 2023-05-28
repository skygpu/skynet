#!/usr/bin/python

import json

from typing import Union, Optional
from pathlib import Path
from contextlib import contextmanager as cm

from ..constants import *


class ConfigRequestFormatError(BaseException):
    ...

class ConfigUnknownAttribute(BaseException):
    ...

class ConfigUnknownAlgorithm(BaseException):
    ...

class ConfigUnknownUpscaler(BaseException):
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
                case 'algo':
                    val = params[2]
                    if val not in ALGOS:
                        raise ConfigUnknownAlgorithm(f'no algo named {val}')

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

