#!/usr/bin/python

import logging

from datetime import datetime

import pynng

from telebot.async_telebot import AsyncTeleBot
from trio_asyncio import aio_as_trio

from ..constants import *

from . import *


PREFIX = 'tg'


async def run_skynet_telegram(
    tg_token: str,
    key_name: str = 'telegram-frontend',
    cert_name: str = 'whitelist/telegram-frontend'
):

    logging.basicConfig(level=logging.INFO)
    bot = AsyncTeleBot(tg_token)

    with open_skynet_rpc(
        'skynet-telegram-0',
        security=True,
        cert_name=cert,
        key_name=key
    ) as rpc_call:

        async def _rpc_call(
            uid: int,
            method: str,
            params: dict
        ):
            return await rpc_call(
                method, params, uid=f'{PREFIX}+{uid}')

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
            rpc_params = {}
            try:
                attr, val, reply_txt = validate_user_config_request(
                    message.text)

                resp = await _rpc_call(
                    message.from_user.id,
                    'config', {'attr': attr, 'val': val})

            except BaseException as e:
                reply_text = str(e.value)

            finally:
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
