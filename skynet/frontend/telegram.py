#!/usr/bin/python

import io
import zlib
import random
import logging
import asyncio
import traceback

from decimal import Decimal
from hashlib import sha256
from datetime import datetime, timedelta

import asks
import docker

from PIL import Image
from leap.cleos import CLEOS, default_nodeos_image
from leap.sugar import get_container, collect_stdout
from leap.hyperion import HyperionAPI
from trio_asyncio import aio_as_trio
from telebot.types import (
    InputFile, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
)

from telebot.types import CallbackQuery
from telebot.async_telebot import AsyncTeleBot, ExceptionHandler
from telebot.formatting import hlink

from ..db import open_new_database, open_database_connection
from ..ipfs import open_ipfs_node, get_ipfs_file
from ..constants import *

from . import *


class SKYExceptionHandler(ExceptionHandler):

    def handle(exception):
        traceback.print_exc()


def build_redo_menu():
    btn_redo = InlineKeyboardButton("Redo", callback_data=json.dumps({'method': 'redo'}))
    inline_keyboard = InlineKeyboardMarkup()
    inline_keyboard.add(btn_redo)
    return inline_keyboard


def prepare_metainfo_caption(tguser, worker: str, reward: str, meta: dict) -> str:
    prompt = meta["prompt"]
    if len(prompt) > 256:
        prompt = prompt[:256]

    if tguser.username:
        user = f'@{tguser.username}'
    else:
        user = f'{tguser.first_name} id: {tguser.id}'

    meta_str = f'<u>by {user}</u>\n'
    meta_str += f'<i>performed by {worker}</i>\n'
    meta_str += f'<b><u>reward: {reward}</u></b>\n'

    meta_str += f'<code>prompt:</code> {prompt}\n'
    meta_str += f'<code>seed: {meta["seed"]}</code>\n'
    meta_str += f'<code>step: {meta["step"]}</code>\n'
    meta_str += f'<code>guidance: {meta["guidance"]}</code>\n'
    if meta['strength']:
        meta_str += f'<code>strength: {meta["strength"]}</code>\n'
    meta_str += f'<code>algo: {meta["algo"]}</code>\n'
    if meta['upscaler']:
        meta_str += f'<code>upscaler: {meta["upscaler"]}</code>\n'

    meta_str += f'<b><u>Made with Skynet v{VERSION}</u></b>\n'
    meta_str += f'<b>JOIN THE SWARM: @skynetgpu</b>'
    return meta_str


def generate_reply_caption(
    tguser,  # telegram user
    params: dict,
    ipfs_hash: str,
    tx_hash: str,
    worker: str,
    reward: str
):
    ipfs_link = hlink(
        'Get your image on IPFS',
        f'https://ipfs.ancap.tech/ipfs/{ipfs_hash}/image.png'
    )
    explorer_link = hlink(
        'SKYNET Transaction Explorer',
        f'https://skynet.ancap.tech/v2/explore/transaction/{tx_hash}'
    )

    meta_info = prepare_metainfo_caption(tguser, worker, reward, params)

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
    message, user, chat,
    account: str,
    permission: str,
    params: dict,
    file_id: str | None = None,
    binary_data: str = ''
):
    if params['seed'] == None:
        params['seed'] = random.randint(0, 0xFFFFFFFF)

    sanitized_params = {}
    for key, val in params.items():
        if isinstance(val, Decimal):
            val = int(val)

        sanitized_params[key] = val

    body = json.dumps({
        'method': 'diffuse',
        'params': sanitized_params
    })
    request_time = datetime.now().isoformat()

    reward = '20.0000 GPU'
    res = await cleos.s_push_action(
        'telos.gpu',
        'enqueue',
        {
            'user': Name(account),
            'request_body': body,
            'binary_data': binary_data,
            'reward': asset_from_str(reward)
        },
        account, key, permission=permission
    )

    if 'code' in res:
        await bot.reply_to(message, json.dumps(res, indent=4))
        return

    out = collect_stdout(res)

    request_id, nonce = out.split(':')

    request_hash = sha256(
        (nonce + body + binary_data).encode('utf-8')).hexdigest().upper()

    request_id = int(request_id)
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
            data = actions[0]['act']['data']
            ipfs_hash = data['ipfs_hash']
            worker = data['worker']
            logging.info('Found matching submit!')
            break

        await asyncio.sleep(1)

    if not ipfs_hash:
        await bot.reply_to(message, 'timeout processing request')
        return

    # attempt to get the image and send it
    ipfs_link = f'http://test1.us.telos.net:8080/ipfs/{ipfs_hash}/image.png'
    resp = await get_ipfs_file(ipfs_link)

    caption = generate_reply_caption(
        user, params, ipfs_hash, tx_hash, worker, reward)

    if not resp or resp.status_code != 200:
        logging.error(f'couldn\'t get ipfs hosted image at {ipfs_link}!')
        await bot.reply_to(
            message,
            caption,
            reply_markup=build_redo_menu(),
            parse_mode='HTML'
        )

    else:
        logging.info(f'succes! sending generated image')
        if file_id:  # img2img
            await bot.send_media_group(
                chat.id,
                media=[
                    InputMediaPhoto(file_id),
                    InputMediaPhoto(
                        resp.raw,
                        caption=caption,
                        parse_mode='HTML'
                    )
                ],
            )

        else:  # txt2img
            await bot.send_photo(
                chat.id,
                caption=caption,
                photo=resp.raw,
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
    remote_ipfs_node: str,
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

    bot = AsyncTeleBot(tg_token, exception_handler=SKYExceptionHandler)
    logging.info(f'tg_token: {tg_token}')

    with open_ipfs_node() as ipfs_node:
        ipfs_node.connect(remote_ipfs_node)
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

            async def _generic_txt2img(message_or_query):
                if isinstance(message_or_query, CallbackQuery):
                    query = message_or_query
                    message = query.message
                    user = query.from_user
                    chat = query.message.chat

                else:
                    message = message_or_query
                    user = message.from_user
                    chat = message.chat

                reply_id = None
                if chat.type == 'group' and chat.id == GROUP_ID:
                    reply_id = message.message_id

                prompt = ' '.join(message.text.split(' ')[1:])

                if len(prompt) == 0:
                    await bot.reply_to(message, 'Empty text prompt ignored.')
                    return

                logging.info(f'mid: {message.id}')

                user_row = await db_call('get_or_create_user', user.id)
                user_config = {**user_row}
                del user_config['id']

                params = {
                    'prompt': prompt,
                    **user_config
                }

                await db_call(
                    'update_user_stats', user.id, 'txt2img', last_prompt=prompt)

                ec = await work_request(
                    bot, cleos, hyperion,
                    message, user, chat,
                    account, permission, params
                )

                if ec == 0:
                    await db_call('increment_generated', user.id)

            async def _generic_img2img(message_or_query):
                if isinstance(message_or_query, CallbackQuery):
                    query = message_or_query
                    message = query.message
                    user = query.from_user
                    chat = query.message.chat

                else:
                    message = message_or_query
                    user = message.from_user
                    chat = message.chat

                reply_id = None
                if chat.type == 'group' and chat.id == GROUP_ID:
                    reply_id = message.message_id

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
                image_raw = await bot.download_file(file_path)

                with Image.open(io.BytesIO(image_raw)) as image:
                    w, h = image.size

                    if w > 512 or h > 512:
                        logging.warning(f'user sent img of size {image.size}')
                        image.thumbnail((512, 512))
                        logging.warning(f'resized it to {image.size}')

                    image.save(f'ipfs-docker-staging/image.png', format='PNG')

                    ipfs_hash = ipfs_node.add('image.png')
                    ipfs_node.pin(ipfs_hash)

                    logging.info(f'published input image {ipfs_hash} on ipfs')

                logging.info(f'mid: {message.id}')

                user_row = await db_call('get_or_create_user', user.id)
                user_config = {**user_row}
                del user_config['id']

                params = {
                    'prompt': prompt,
                    **user_config
                }

                await db_call(
                    'update_user_stats',
                    user.id,
                    'img2img',
                    last_file=file_id,
                    last_prompt=prompt,
                    last_binary=ipfs_hash
                )

                ec = await work_request(
                    bot, cleos, hyperion,
                    message, user, chat,
                    account, permission, params,
                    file_id=file_id,
                    binary_data=ipfs_hash
                )

                if ec == 0:
                    await db_call('increment_generated', user.id)

            @bot.message_handler(commands=['txt2img'])
            async def send_txt2img(message):
                await _generic_txt2img(message)

            @bot.message_handler(func=lambda message: True, content_types=[
                'photo', 'document'])
            async def send_img2img(message):
                await _generic_img2img(message)

            @bot.message_handler(commands=['img2img'])
            async def img2img_missing_image(message):
                await bot.reply_to(
                    message,
                    'seems you tried to do an img2img command without sending image'
                )

            async def _redo(message_or_query):
                if isinstance(message_or_query, CallbackQuery):
                    query = message_or_query
                    message = query.message
                    user = query.from_user
                    chat = query.message.chat

                else:
                    message = message_or_query
                    user = message.from_user
                    chat = message.chat

                method = await db_call('get_last_method_of', user.id)
                prompt = await db_call('get_last_prompt_of', user.id)

                file_id = None
                binary = ''
                if method == 'img2img':
                    file_id = await db_call('get_last_file_of', user.id)
                    binary = await db_call('get_last_binary_of', user.id)

                if not prompt:
                    await bot.reply_to(
                        message,
                        'no last prompt found, do a txt2img cmd first!'
                    )
                    return


                user_row = await db_call('get_or_create_user', user.id)
                user_config = {**user_row}
                del user_config['id']

                params = {
                    'prompt': prompt,
                    **user_config
                }

                await work_request(
                    bot, cleos, hyperion,
                    message, user, chat,
                    account, permission, params,
                    file_id=file_id,
                    binary_data=binary
                )

            @bot.message_handler(commands=['queue'])
            async def queue(message):
                an_hour_ago = datetime.now() - timedelta(hours=1)
                queue = await cleos.aget_table(
                    'telos.gpu', 'telos.gpu', 'queue',
                    index_position=2,
                    key_type='i64',
                    sort='desc',
                    lower_bound=int(an_hour_ago.timestamp())
                )
                await bot.reply_to(
                    message, f'Total requests on skynet queue: {len(queue)}')

            @bot.message_handler(commands=['redo'])
            async def redo(message):
                await _redo(message)

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

                await db_call('get_or_create_user', user)
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
