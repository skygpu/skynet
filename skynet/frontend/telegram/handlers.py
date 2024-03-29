#!/usr/bin/python

import io
import json
import logging

from datetime import datetime, timedelta

from PIL import Image
from telebot.types import CallbackQuery, Message

from skynet.frontend import validate_user_config_request, perform_auto_conf
from skynet.constants import *


def create_handler_context(frontend: 'SkynetTelegramFrontend'):

    bot = frontend.bot
    cleos = frontend.cleos
    db_call = frontend.db_call
    work_request = frontend.work_request

    ipfs_node = frontend.ipfs_node

    # generic / simple handlers

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


    # generic txt2img handler

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

        if chat.type == 'private':
            return

        reply_id = None
        if chat.type == 'group' and chat.id == GROUP_ID:
            reply_id = message.message_id

        user_row = await db_call('get_or_create_user', user.id)

        # init new msg
        init_msg = 'started processing txt2img request...'
        status_msg = await bot.reply_to(message, init_msg)
        await db_call(
            'new_user_request', user.id, message.id, status_msg.id, status=init_msg)

        prompt = ' '.join(message.text.split(' ')[1:])

        if len(prompt) == 0:
            await bot.edit_message_text(
                'Empty text prompt ignored.',
                chat_id=status_msg.chat.id,
                message_id=status_msg.id
            )
            await db_call('update_user_request', status_msg.id, 'Empty text prompt ignored.')
            return

        logging.info(f'mid: {message.id}')

        user_config = {**user_row}
        del user_config['id']

        if user_config['autoconf']:
            user_config = perform_auto_conf(user_config)

        params = {
            'prompt': prompt,
            **user_config
        }

        await db_call(
            'update_user_stats', user.id, 'txt2img', last_prompt=prompt)

        success = await work_request(user, status_msg, 'txt2img', params)

        if success:
            await db_call('increment_generated', user.id)


    # generic img2img handler

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

        if chat.type == 'private':
            return

        reply_id = None
        if chat.type == 'group' and chat.id == GROUP_ID:
            reply_id = message.message_id

        user_row = await db_call('get_or_create_user', user.id)

        # init new msg
        init_msg = 'started processing txt2img request...'
        status_msg = await bot.reply_to(message, init_msg)
        await db_call(
            'new_user_request', user.id, message.id, status_msg.id, status=init_msg)

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

        user_config = {**user_row}
        del user_config['id']
        if user_config['autoconf']:
            user_config = perform_auto_conf(user_config)

        with Image.open(io.BytesIO(image_raw)) as image:
            w, h = image.size

            if w > user_config['width'] or h > user_config['height']:
                logging.warning(f'user sent img of size {image.size}')
                image.thumbnail(
                    (user_config['width'], user_config['height']))
                logging.warning(f'resized it to {image.size}')

            image_loc = 'ipfs-staging/image.png'
            image.save(image_loc, format='PNG')

            ipfs_info = await ipfs_node.add(image_loc)
            ipfs_hash = ipfs_info['Hash']
            await ipfs_node.pin(ipfs_hash)

            logging.info(f'published input image {ipfs_hash} on ipfs')

        logging.info(f'mid: {message.id}')

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

        success = await work_request(
            user, status_msg, 'img2img', params,
            file_id=file_id,
            binary_data=ipfs_hash
        )

        if success:
            await db_call('increment_generated', user.id)


    # generic redo handler

    async def _redo(message_or_query):
        is_query = False
        if isinstance(message_or_query, CallbackQuery):
            is_query = True
            query = message_or_query
            message = query.message
            user = query.from_user
            chat = query.message.chat

        elif isinstance(message_or_query, Message):
            message = message_or_query
            user = message.from_user
            chat = message.chat

        if chat.type == 'private':
            return

        init_msg = 'started processing redo request...'
        if is_query:
            status_msg = await bot.send_message(chat.id, init_msg)

        else:
            status_msg = await bot.reply_to(message, init_msg)

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
        await db_call(
            'new_user_request', user.id, message.id, status_msg.id, status=init_msg)
        user_config = {**user_row}
        del user_config['id']
        if user_config['autoconf']:
            user_config = perform_auto_conf(user_config)

        params = {
            'prompt': prompt,
            **user_config
        }

        success = await work_request(
            user, status_msg, 'redo', params,
            file_id=file_id,
            binary_data=binary
        )

        if success:
            await db_call('increment_generated', user.id)


    # "proxy" handlers just request routers

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

    @bot.message_handler(commands=['redo'])
    async def redo(message):
        await _redo(message)

    @bot.callback_query_handler(func=lambda call: True)
    async def callback_query(call):
        msg = json.loads(call.data)
        logging.info(call.data)
        method = msg.get('method')
        match method:
            case 'redo':
                await _redo(call)


    # catch all handler for things we dont support

    @bot.message_handler(func=lambda message: True)
    async def echo_message(message):
        if message.text[0] == '/':
            await bot.reply_to(message, UNKNOWN_CMD_TEXT)
