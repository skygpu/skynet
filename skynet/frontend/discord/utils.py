#!/usr/bin/python

import json
import logging
import traceback

from datetime import datetime, timezone

from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from telebot.async_telebot import ExceptionHandler
from telebot.formatting import hlink
import discord

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


def prepare_metainfo_caption(user, worker: str, reward: str, meta: dict) -> str:
    prompt = meta["prompt"]
    if len(prompt) > 256:
        prompt = prompt[:256]

    meta_str = f'__by {user.name}__\n'
    meta_str += f'*performed by {worker}*\n'
    meta_str += f'__**reward: {reward}**__\n'

    meta_str += f'`prompt:` {prompt}\n'
    meta_str += f'`seed: {meta["seed"]}`\n'
    meta_str += f'`step: {meta["step"]}`\n'
    meta_str += f'`guidance: {meta["guidance"]}`\n'
    if meta['strength']:
        meta_str += f'`strength: {meta["strength"]}`\n'
    meta_str += f'`algo: {meta["model"]}`\n'
    if meta['upscaler']:
        meta_str += f'`upscaler: {meta["upscaler"]}`\n'

    meta_str += f'__**Made with Skynet v{VERSION}**__\n'
    meta_str += f'**JOIN THE SWARM: @skynetgpu**'
    return meta_str


def generate_reply_caption(
    user,  # discord user
    params: dict,
    tx_hash: str,
    worker: str,
    reward: str
):
    explorer_link = discord.Embed(
        title='[SKYNET Transaction Explorer]',
        url=f'https://explorer.{DEFAULT_DOMAIN}/v2/explore/transaction/{tx_hash}',
        color=discord.Color.blue())

    meta_info = prepare_metainfo_caption(user, worker, reward, params)

    # why do we have this?
    final_msg = '\n'.join([
        'Worker finished your task!',
        # explorer_link,
        f'PARAMETER INFO:\n{meta_info}'
    ])

    final_msg = '\n'.join([
        # f'***{explorer_link}***',
        f'{meta_info}'
    ])

    logging.info(final_msg)

    return final_msg, explorer_link


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
