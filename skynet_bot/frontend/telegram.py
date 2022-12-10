#!/usr/bin/python

import logging

from datetime import datetime

import pynng

from telebot.async_telebot import AsyncTeleBot
from trio_asyncio import aio_as_trio

from ..constants import *

from . import *


PREFIX = 'tg'


async def run_skynet_telegram(tg_token: str):

    logging.basicConfig(level=logging.INFO)
    bot = AsyncTeleBot(tg_token)

    with open_skynet_rpc() as rpc_sock:

        async def _rpc_call(
            uid: int,
            method: str,
            params: dict
        ):
            return await rpc_call(
                rpc_sock, f'{PREFIX}+{uid}', method, params)

        @bot.message_handler(commands=['help'])
        async def send_help(message):
            await bot.reply_to(message, HELP_TEXT)

        @bot.message_handler(commands=['cool'])
        async def send_cool_words(message):
            await bot.reply_to(message, '\n'.join(COOL_WORDS))

        @bot.message_handler(commands=['txt2img'])
        async def send_txt2img(message):
            resp = await _rpc_call(
                message.from_user.id,
                'txt2img',
                {}
            )

        @bot.message_handler(commands=['redo'])
        async def redo_txt2img(message):
            resp = await _rpc_call(
                message.from_user.id,
                'redo',
                {}
            )

        @bot.message_handler(commands=['config'])
        async def set_config(message):
            params = message.text.split(' ')

            rpc_params = {}

            if len(params) < 3:
                bot.reply_to(message, 'wrong msg format')

            else:

                try:
                    attr = params[1]

                    if attr == 'algo':
                        val = params[2]
                        if val not in ALGOS:
                            raise ConfigUnknownAlgorithm

                    elif attr == 'step':
                        val = int(params[2])
                        val = max(min(val, MAX_STEP), MIN_STEP)

                    elif attr  == 'width':
                        val = max(min(int(params[2]), MAX_WIDTH), 16)
                        if val % 8 != 0:
                            raise ConfigSizeDivisionByEight

                    elif attr  == 'height':
                        val = max(min(int(params[2]), MAX_HEIGHT), 16)
                        if val % 8 != 0:
                            raise ConfigSizeDivisionByEight

                    elif attr == 'seed':
                        val = params[2]
                        if val == 'auto':
                            val = None
                        else:
                            val = int(params[2])

                    elif attr == 'guidance':
                        val = float(params[2])
                        val = max(min(val, MAX_GUIDANCE), 0)

                    elif attr == 'upscaler':
                        val = params[2]
                        if val == 'off':
                            val = None
                        elif val != 'x4':
                            raise ConfigUnknownUpscaler

                    else:
                        raise ConfigUnknownAttribute

                    resp = await _rpc_call(
                        message.from_user.id,
                        'config', {'attr': attr, 'val': val})

                    reply_txt = f'config updated! {attr} to {val}'

                except ConfigUnknownAlgorithm:
                    reply_txt = f'no algo named {val}'

                except ConfigUnknownAttribute:
                    reply_txt = f'\"{attr}\" not a configurable parameter'

                except ConfigUnknownUpscaler:
                    reply_txt = f'\"{val}\" is not a valid upscaler'

                except ConfigSizeDivisionByEight:
                    reply_txt = 'size must be divisible by 8!'

                except ValueError:
                    reply_txt = f'\"{val}\" is not a number silly'

                await bot.reply_to(message, reply_txt)

        @bot.message_handler(commands=['stats'])
        async def user_stats(message):
            resp = await _rpc_call(
                message.from_user.id,
                'stats',
                {}
            )
            stats = resp.result

            stats_str = f'generated: {stats["generated"]}\n'
            stats_str += f'joined: {stats["joined"]}\n'
            stats_str += f'role: {stats["role"]}\n'

            await bot.reply_to(
                message, stats_str)

        @bot.message_handler(commands=['donate'])
        async def donation_info(message):
            await bot.reply_to(
                message, DONATION_INFO)


        @bot.message_handler(func=lambda message: True)
        async def echo_message(message):
            if message.text[0] == '/':
                await bot.reply_to(message, UNKNOWN_CMD_TEXT)


        await aio_as_trio(bot.infinity_polling())
