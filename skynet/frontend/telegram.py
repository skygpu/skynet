#!/usr/bin/python

import io
import zlib
import logging

from datetime import datetime

from PIL import Image
from trio_asyncio import aio_as_trio

from telebot.types import (
    InputFile, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
)
from telebot.async_telebot import AsyncTeleBot

from ..db import open_database_connection
from ..constants import *

from . import *


PREFIX = 'tg'

def build_redo_menu():
    btn_redo = InlineKeyboardButton("Redo", callback_data=json.dumps({'method': 'redo'}))
    inline_keyboard = InlineKeyboardMarkup()
    inline_keyboard.add(btn_redo)
    return inline_keyboard


def prepare_metainfo_caption(tguser, meta: dict) -> str:
    prompt = meta["prompt"]
    if len(prompt) > 256:
        prompt = prompt[:256]

    if tguser.username:
        user = f'@{tguser.username}'
    else:
        user = f'{tguser.first_name} id: {tguser.id}'

    meta_str = f'by {user}\n'
    meta_str += f'prompt: \"{prompt}\"\n'
    meta_str += f'seed: {meta["seed"]}\n'
    meta_str += f'step: {meta["step"]}\n'
    meta_str += f'guidance: {meta["guidance"]}\n'
    if meta['strength']:
        meta_str += f'strength: {meta["strength"]}\n'
    meta_str += f'algo: \"{meta["algo"]}\"\n'
    if meta['upscaler']:
        meta_str += f'upscaler: \"{meta["upscaler"]}\"\n'
    meta_str += f'sampler: k_euler_ancestral\n'
    meta_str += f'skynet v{VERSION}'
    return meta_str


async def run_skynet_telegram(
    name: str,
    tg_token: str,
    key_name: str = 'telegram-frontend.key',
    cert_name: str = 'whitelist/telegram-frontend.cert',
    rpc_address: str = DEFAULT_RPC_ADDR,
    db_host: str = 'localhost:5432',
    db_user: str = 'skynet',
    db_pass: str = 'password'
):

    logging.basicConfig(level=logging.INFO)
    bot = AsyncTeleBot(tg_token)
    logging.info(f'tg_token: {tg_token}')

    async with open_database_connection(
        db_user, db_pass, db_host
    ) as db_call:
        with open_skynet_rpc(
            f'skynet-telegram-{name}',
            rpc_address=rpc_address,
            cert_name=cert_name,
            key_name=key_name
        ) as session:

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
                chat = message.chat
                reply_id = None
                if chat.type == 'group' and chat.id == GROUP_ID:
                    reply_id = message.message_id

                user_id = f'tg+{message.from_user.id}'

                prompt = ' '.join(message.text.split(' ')[1:])

                if len(prompt) == 0:
                    await bot.reply_to(message, 'Empty text prompt ignored.')
                    return

                logging.info(f'mid: {message.id}')
                user = await db_call('get_or_create_user', user_id)
                user_config = {**(await db_call('get_user_config', user))}
                del user_config['id']

                resp = await session.rpc(
                    'dgpu_call', {
                        'method': 'diffuse',
                        'params': {
                            'prompt': prompt,
                            **user_config
                        }
                    },
                    timeout=60
                )
                logging.info(f'resp to {message.id} arrived')

                resp_txt = ''
                result = MessageToDict(resp.result)
                if 'error' in resp.result:
                    resp_txt = resp.result['message']
                    await bot.reply_to(message, resp_txt)

                else:
                    logging.info(result['id'])
                    img_raw = resp.bin
                    logging.info(f'got image of size: {len(img_raw)}')
                    img = Image.open(io.BytesIO(img_raw))

                    await bot.send_photo(
                        GROUP_ID,
                        caption=prepare_metainfo_caption(message.from_user, result['meta']['meta']),
                        photo=img,
                        reply_to_message_id=reply_id,
                        reply_markup=build_redo_menu()
                    )
                    return


            @bot.message_handler(func=lambda message: True, content_types=['photo'])
            async def send_img2img(message):
                chat = message.chat
                reply_id = None
                if chat.type == 'group' and chat.id == GROUP_ID:
                    reply_id = message.message_id

                user_id = f'tg+{message.from_user.id}'

                if not message.caption.startswith('/img2img'):
                    await bot.reply_to(
                        message,
                        'For image to image you need to add /img2img to the beggining of your caption'
                    )
                    return

                prompt = ' '.join(message.caption.split(' ')[1:])

                if len(prompt) == 0:
                    await bot.reply_to(message, 'Empty text prompt ignored.')
                    return

                file_id = message.photo[-1].file_id
                file_path = (await bot.get_file(file_id)).file_path
                file_raw = await bot.download_file(file_path)

                logging.info(f'mid: {message.id}')

                user = await db_call('get_or_create_user', user_id)
                user_config = {**(await db_call('get_user_config', user))}
                del user_config['id']

                resp = await session.rpc(
                    'dgpu_call', {
                        'method': 'diffuse',
                        'params': {
                            'prompt': prompt,
                            **user_config
                        }
                    },
                    binext=file_raw,
                    timeout=60
                )
                logging.info(f'resp to {message.id} arrived')

                resp_txt = ''
                result = MessageToDict(resp.result)
                if 'error' in resp.result:
                    resp_txt = resp.result['message']
                    await bot.reply_to(message, resp_txt)

                else:
                    logging.info(result['id'])
                    img_raw = resp.bin
                    logging.info(f'got image of size: {len(img_raw)}')
                    img = Image.open(io.BytesIO(img_raw))

                    await bot.send_media_group(
                        GROUP_ID,
                        media=[
                            InputMediaPhoto(file_id),
                            InputMediaPhoto(
                                img,
                                caption=prepare_metainfo_caption(message.from_user, result['meta']['meta'])
                            )
                        ],
                        reply_to_message_id=reply_id
                    )
                    return


            @bot.message_handler(commands=['img2img'])
            async def img2img_missing_image(message):
                await bot.reply_to(
                    message,
                    'seems you tried to do an img2img command without sending image'
                )

            @bot.message_handler(commands=['redo'])
            async def redo(message):
                chat = message.chat
                reply_id = None
                if chat.type == 'group' and chat.id == GROUP_ID:
                    reply_id = message.message_id

                user_config = {**(await db_call('get_user_config', user))}
                del user_config['id']
                prompt = await db_call('get_last_prompt_of', user)

                resp = await session.rpc(
                    'dgpu_call', {
                        'method': 'diffuse',
                        'params': {
                            'prompt': prompt,
                            **user_config
                        }
                    },
                    timeout=60
                )
                logging.info(f'resp to {message.id} arrived')

                resp_txt = ''
                result = MessageToDict(resp.result)
                if 'error' in resp.result:
                    resp_txt = resp.result['message']
                    await bot.reply_to(message, resp_txt)

                else:
                    logging.info(result['id'])
                    img_raw = resp.bin
                    logging.info(f'got image of size: {len(img_raw)}')
                    img = Image.open(io.BytesIO(img_raw))

                    await bot.send_photo(
                        GROUP_ID,
                        caption=prepare_metainfo_caption(message.from_user, result['meta']['meta']),
                        photo=img,
                        reply_to_message_id=reply_id
                    )
                    return

            @bot.message_handler(commands=['config'])
            async def set_config(message):
                rpc_params = {}
                try:
                    attr, val, reply_txt = validate_user_config_request(
                        message.text)

                    logging.info(f'user config update: {attr} to {val}')
                    await db_call('update_user_config',
                        user, req.params['attr'], req.params['val'])
                    logging.info('done')

                except BaseException as e:
                    reply_txt = str(e)

                finally:
                    await bot.reply_to(message, reply_txt)

            @bot.message_handler(commands=['stats'])
            async def user_stats(message):

                generated, joined, role = await db_call('get_user_stats', user)

                stats_str = f'generated: {generated}\n'
                stats_str += f'joined: {joined}\n'
                stats_str += f'role: {role}\n'

                await bot.reply_to(
                    message, stats_str)

            @bot.message_handler(commands=['donate'])
            async def donation_info(message):
                await bot.reply_to(
                    message, DONATION_INFO)

            @bot.message_handler(commands=['say'])
            async def say(message):
                chat = message.chat
                user = message.from_user

                if (chat.type == 'group') or (user.id != 383385940):
                    return

                await bot.send_message(GROUP_ID, message.text[4:])


            @bot.message_handler(func=lambda message: True)
            async def echo_message(message):
                if message.text[0] == '/':
                    await bot.reply_to(message, UNKNOWN_CMD_TEXT)

        @bot.callback_query_handler(func=lambda call: True)
        async def callback_query(call):
            msg = json.loads(call.data)
            logging.info(call.data)
            method = msg.get('method')
            match method:
                case 'redo':
                    await _redo(call)


            await aio_as_trio(bot.infinity_polling)()
