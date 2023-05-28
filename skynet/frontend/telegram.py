#!/usr/bin/python

import io
import zlib
import logging
import asyncio

from hashlib import sha256
from datetime import datetime

import docker

from PIL import Image
from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, collect_stdout
from leap.hyperion import HyperionAPI
from trio_asyncio import aio_as_trio
from telebot.types import (
    InputFile, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
)
from telebot.async_telebot import AsyncTeleBot
from telebot.formatting import hlink

from ..db import open_new_database, open_database_connection
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

    meta_str = f'<u>by {user}</u>\n'

    meta_str += f'<code>prompt:</code> {prompt}\n'
    meta_str += f'<code>seed: {meta["seed"]}</code>\n'
    meta_str += f'<code>step: {meta["step"]}</code>\n'
    meta_str += f'<code>guidance: {meta["guidance"]}</code>\n'
    if meta['strength']:
        meta_str += f'<code>strength: {meta["strength"]}</code>\n'
    meta_str += f'<code>algo: {meta["algo"]}</code>\n'
    if meta['upscaler']:
        meta_str += f'<code>upscaler: {meta["upscaler"]}</code>\n'

    meta_str += f'<b><u>Made with Skynet {VERSION}</u></b>\n'
    meta_str += f'<b>JOIN THE SWARM: @skynetgpu</b>'
    return meta_str


def generate_reply_caption(
    tguser,  # telegram user
    params: dict,
    ipfs_hash: str,
    tx_hash: str
):
    ipfs_link = hlink(
        'Get your image on IPFS',
        f'http://test1.us.telos.net:8080/ipfs/{ipfs_hash}/image.png'
    )
    explorer_link = hlink(
        'SKYNET Transaction Explorer',
        f'http://test1.us.telos.net:42001/v2/explore/transaction/{tx_hash}'
    )

    meta_info = prepare_metainfo_caption(tguser, params)

    final_msg = '\n'.join([
        'Worker finished your task!',
        ipfs_link,
        explorer_link,
        f'PARAMETER INFO:\n{meta_info}'
    ])

    final_msg = '\n'.join([
        f'<b>{ipfs_link}</b>',
        f'<i>{explorer_link}</i>',
        f'{meta_info}'
    ])

    logging.info(final_msg)

    return final_msg


async def get_global_config(cleos):
    return (await cleos.aget_table(
        'telos.gpu', 'telos.gpu', 'config'))[0]

async def get_user_nonce(cleos, user: str):
    return (await cleos.aget_table(
        'telos.gpu', 'telos.gpu', 'users',
        index_position=1,
        key_type='name',
        lower_bound=user,
        upper_bound=user
    ))[0]['nonce']

async def work_request(
    bot, cleos, hyperion,
    message,
    account: str,
    permission: str,
    params: dict
):
    body = json.dumps({
        'method': 'diffuse',
        'params': params
    })
    user = message.from_user
    chat = message.chat
    request_time = datetime.now().isoformat()
    ec, out = cleos.push_action(
        'telos.gpu', 'enqueue', [account, body, '', '20.0000 GPU'], f'{account}@{permission}'
    )
    out = collect_stdout(out)
    if ec != 0:
        await bot.reply_to(message, out)
        return

    nonce = await get_user_nonce(cleos, account)
    request_hash = sha256(
        (str(nonce) + body).encode('utf-8')).hexdigest().upper()

    request_id = int(out)
    logging.info(f'{request_id} enqueued.')

    config = await get_global_config(cleos)

    tx_hash = None
    ipfs_hash = None
    for i in range(60):
        submits = await hyperion.aget_actions(
            account=account,
            filter='telos.gpu:submit',
            sort='desc',
            after=request_time
        )
        actions = [
            action
            for action in submits['actions']
            if action[
                'act']['data']['request_hash'] == request_hash
        ]
        if len(actions) > 0:
            tx_hash = actions[0]['trx_id']
            ipfs_hash = actions[0]['act']['data']['ipfs_hash']
            break

        await asyncio.sleep(1)

    if not ipfs_hash:
        await bot.reply_to(message, 'timeout processing request')
        return

    await bot.reply_to(
        message,
        generate_reply_caption(
            user, params, ipfs_hash, tx_hash),
        reply_markup=build_redo_menu(),
        parse_mode='HTML'
    )


async def run_skynet_telegram(
    tg_token: str,
    account: str,
    permission: str,
    node_url: str,
    hyperion_url: str,
    db_host: str,
    db_user: str,
    db_pass: str,
    key: str = None
):
    dclient = docker.from_env()
    vtestnet = get_container(
        dclient,
        default_nodeos_image(),
        force_unique=True,
        detach=True,
        network='host',
        remove=True)

    cleos = CLEOS(dclient, vtestnet, url=node_url, remote=node_url)
    hyperion = HyperionAPI(hyperion_url)

    logging.basicConfig(level=logging.INFO)

    if key:
        cleos.setup_wallet(key)

    bot = AsyncTeleBot(tg_token)
    logging.info(f'tg_token: {tg_token}')

    async with open_database_connection(
        db_user, db_pass, db_host
    ) as db_call:

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
            user = message.from_user.id
            chat = message.chat
            reply_id = None
            if chat.type == 'group' and chat.id == GROUP_ID:
                reply_id = message.message_id

            prompt = ' '.join(message.text.split(' ')[1:])

            if len(prompt) == 0:
                await bot.reply_to(message, 'Empty text prompt ignored.')
                return

            logging.info(f'mid: {message.id}')

            await db_call('get_or_create_user', user)
            user_config = {**(await db_call('get_user_config', user))}
            del user_config['id']

            params = {
                'prompt': prompt,
                **user_config
            }

            await db_call('update_user_stats', user, last_prompt=prompt)

            await work_request(
                bot, cleos, hyperion,
                message, account, permission, params)

        @bot.message_handler(func=lambda message: True, content_types=['photo'])
        async def send_img2img(message):
            user = message.from_user.id
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

            req = json.dumps({
                'method': 'diffuse',
                'params': {
                    'prompt': prompt,
                    **user_config
                }
            })

            ec, out = cleos.push_action(
                'telos.gpu', 'enqueue', [account, req, file_raw.hex()], f'{account}@{permission}'
            )
            if ec != 0:
                await bot.reply_to(message, out)
                return

            request_id = int(out)
            logging.info(f'{request_id} enqueued.')

            ipfs_hash = None
            sha_hash = None
            for i in range(60):
                result = cleos.get_table(
                    'telos.gpu', 'telos.gpu', 'results',
                    index_position=2,
                    key_type='i64',
                    lower_bound=request_id,
                    upper_bound=request_id
                )
                if len(results) > 0:
                    ipfs_hash = result[0]['ipfs_hash']
                    sha_hash = result[0]['result_hash']
                    break
                else:
                    await asyncio.sleep(1)

            if not ipfs_hash:
                await bot.reply_to(message, 'timeout processing request')

            ipfs_link = f'https://ipfs.io/ipfs/{ipfs_hash}/image.png'

            await bot.reply_to(
                message,
                ipfs_link + '\n' +
                prepare_metainfo_caption(user, result['meta']['meta']),
                reply_to_message_id=reply_id,
                reply_markup=build_redo_menu()
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
            user = message.from_user.id
            chat = message.chat
            reply_id = None
            if chat.type == 'group' and chat.id == GROUP_ID:
                reply_id = message.message_id

            user_config = {**(await db_call('get_user_config', user))}
            del user_config['id']
            prompt = await db_call('get_last_prompt_of', user)

            if not prompt:
                await bot.reply_to(
                    message,
                    'no last prompt found, do a txt2img cmd first!'
                )
                return

            params = {
                'prompt': prompt,
                **user_config
            }

            await work_request(
                bot, cleos, hyperion,
                message, account, permission, params)

        @bot.message_handler(commands=['config'])
        async def set_config(message):
            user = message.from_user.id
            try:
                attr, val, reply_txt = validate_user_config_request(
                    message.text)

                logging.info(f'user config update: {attr} to {val}')
                await db_call('update_user_config', user, attr, val)
                logging.info('done')

            except BaseException as e:
                reply_txt = str(e)

            finally:
                await bot.reply_to(message, reply_txt)

        @bot.message_handler(commands=['stats'])
        async def user_stats(message):
            user = message.from_user.id

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

    try:
        await bot.infinity_polling()

    except KeyboardInterrupt:
        ...

    finally:
        vtestnet.stop()
