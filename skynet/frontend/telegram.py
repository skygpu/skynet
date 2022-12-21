#!/usr/bin/python

import io
import base64
import logging

from datetime import datetime

import pynng

from PIL import Image
from trio_asyncio import aio_as_trio

from telebot.types import InputFile
from telebot.async_telebot import AsyncTeleBot

from ..constants import *

from . import *


PREFIX = 'tg'


def prepare_metainfo_caption(meta: dict) -> str:
    meta_str = f'prompt: \"{meta["prompt"]}\"\n'
    meta_str += f'seed: {meta["seed"]}\n'
    meta_str += f'step: {meta["step"]}\n'
    meta_str += f'guidance: {meta["guidance"]}\n'
    meta_str += f'algo: \"{meta["algo"]}\"\n'
    meta_str += f'sampler: k_euler_ancestral\n'
    meta_str += f'skynet v{VERSION}'
    return meta_str


async def run_skynet_telegram(
    tg_token: str,
    key_name: str = 'telegram-frontend',
    cert_name: str = 'whitelist/telegram-frontend'
):

    logging.basicConfig(level=logging.INFO)
    bot = AsyncTeleBot(tg_token)

    async with open_skynet_rpc(
        'skynet-telegram-0',
        security=True,
        cert_name=cert_name,
        key_name=key_name
    ) as rpc_call:

        async def _rpc_call(
            uid: int,
            method: str,
            params: dict = {}
        ):
            return await rpc_call(
                method, params, uid=f'{PREFIX}+{uid}')

        @bot.message_handler(commands=['help'])
        async def send_help(message):
            splt_msg = message.text.split(' ')

            if len(splt_msg) == 1:
                await bot.reply_to(message, HELP_TEXT)

            else:
                param = splt_msg[1]
                if param in HELP_TOPICS:
                    await bot.reply_to(message, HELP_TOPICS[param])

                else:
                    await bot.reply_to(message, HELP_UNKWNOWN_PARAM)

        @bot.message_handler(commands=['cool'])
        async def send_cool_words(message):
            await bot.reply_to(message, '\n'.join(COOL_WORDS))

        @bot.message_handler(commands=['txt2img'])
        async def send_txt2img(message):
            prompt = ' '.join(message.text.split(' ')[1:])

            if len(prompt) == 0:
                await bot.reply_to(message, 'Empty text prompt ignored.')
                return

            logging.info(f'mid: {message.id}')
            resp = await _rpc_call(
                message.from_user.id,
                'txt2img',
                {'prompt': prompt}
            )
            logging.info(f'resp to {message.id} arrived')

            resp_txt = ''
            if 'error' in resp.result:
                resp_txt = resp.result['message']

            else:
                logging.info(resp.result['id'])
                img_raw = base64.b64decode(bytes.fromhex(resp.result['img']))
                img = Image.frombytes('RGB', (512, 512), img_raw)

                await bot.send_photo(
                    message.chat.id,
                    caption=prepare_metainfo_caption(resp.result['meta']),
                    photo=img,
                    reply_to_message_id=message.id
                )
                return

            await bot.reply_to(message, resp_txt)

        @bot.message_handler(commands=['redo'])
        async def redo_txt2img(message):
            resp = await _rpc_call(message.from_user.id, 'redo')

            resp_txt = ''
            if 'error' in resp.result:
                resp_txt = resp.result['message']

            else:
                img_raw = base64.b64decode(bytes.fromhex(resp.result['img']))
                img = Image.frombytes('RGB', (512, 512), img_raw)

                await bot.send_photo(
                    message.chat.id,
                    caption=prepare_metainfo_caption(resp.result['meta']),
                    photo=img,
                    reply_to_message_id=message.id
                )
                return

            await bot.reply_to(message, resp_txt)

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
