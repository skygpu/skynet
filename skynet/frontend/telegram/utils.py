#!/usr/bin/python

import json
import logging
import traceback

from datetime import datetime, timezone

from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from telebot.async_telebot import ExceptionHandler
from telebot.formatting import hlink

from skynet.constants import *


def timestamp_pretty():
    return datetime.now(timezone.utc).strftime('%H:%M:%S')


def tg_user_pretty(tguser):
    if tguser.username:
        return f'@{tguser.username}'
    else:
        return f'{tguser.first_name} id: {tguser.id}'


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


    meta_str = f'<u>by {tg_user_pretty(tguser)}</u>\n'
    meta_str += f'<i>performed by {worker}</i>\n'
    meta_str += f'<b><u>reward: {reward}</u></b>\n'

    meta_str += f'<code>prompt:</code> {prompt}\n'
    meta_str += f'<code>seed: {meta["seed"]}</code>\n'
    meta_str += f'<code>step: {meta["step"]}</code>\n'
    meta_str += f'<code>guidance: {meta["guidance"]}</code>\n'
    if meta['strength']:
        meta_str += f'<code>strength: {meta["strength"]}</code>\n'
    meta_str += f'<code>algo: {meta["model"]}</code>\n'
    if meta['upscaler']:
        meta_str += f'<code>upscaler: {meta["upscaler"]}</code>\n'

    meta_str += f'<b><u>Made with Skynet v{VERSION}</u></b>\n'
    meta_str += f'<b>JOIN THE SWARM: @skynetgpu</b>'
    return meta_str


def generate_reply_caption(
    tguser,  # telegram user
    params: dict,
    tx_hash: str,
    worker: str,
    reward: str,
    explorer_domain: str
):
    explorer_link = hlink(
        'SKYNET Transaction Explorer',
        f'https://{explorer_domain}/v2/explore/transaction/{tx_hash}'
    )

    meta_info = prepare_metainfo_caption(tguser, worker, reward, params)

    final_msg = '\n'.join([
        'Worker finished your task!',
        explorer_link,
        f'PARAMETER INFO:\n{meta_info}'
    ])

    final_msg = '\n'.join([
        f'<b><i>{explorer_link}</i></b>',
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
