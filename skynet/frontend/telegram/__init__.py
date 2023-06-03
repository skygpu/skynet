#!/usr/bin/python

import random
import logging
import asyncio

from decimal import Decimal
from hashlib import sha256
from datetime import datetime
from contextlib import ExitStack, AsyncExitStack
from contextlib import asynccontextmanager as acm

from leap.cleos import CLEOS
from leap.sugar import Name, asset_from_str, collect_stdout
from leap.hyperion import HyperionAPI
from telebot.asyncio_helper import ApiTelegramException
from telebot.types import InputMediaPhoto

from telebot.types import CallbackQuery
from telebot.async_telebot import AsyncTeleBot

from skynet.db import open_new_database, open_database_connection
from skynet.ipfs import open_ipfs_node, get_ipfs_file
from skynet.constants import *

from . import *

from .utils import *
from .handlers import create_handler_context


class SkynetTelegramFrontend:

    def __init__(
        self,
        token: str,
        account: str,
        permission: str,
        node_url: str,
        hyperion_url: str,
        db_host: str,
        db_user: str,
        db_pass: str,
        remote_ipfs_node: str,
        key: str
    ):
        self.token = token
        self.account = account
        self.permission = permission
        self.node_url = node_url
        self.hyperion_url = hyperion_url
        self.db_host = db_host
        self.db_user = db_user
        self.db_pass = db_pass
        self.remote_ipfs_node = remote_ipfs_node
        self.key = key

        self.bot = AsyncTeleBot(token, exception_handler=SKYExceptionHandler)
        self.cleos = CLEOS(None, None, url=node_url, remote=node_url)
        self.hyperion = HyperionAPI(hyperion_url)

        self._exit_stack = ExitStack()
        self._async_exit_stack = AsyncExitStack()

    async def start(self):
        self.ipfs_node = self._exit_stack.enter_context(
            open_ipfs_node())

        self.ipfs_node.connect(self.remote_ipfs_node)
        logging.info(
            f'connected to remote ipfs node: {self.remote_ipfs_node}')

        self.db_call = await self._async_exit_stack.enter_async_context(
            open_database_connection(
                self.db_user, self.db_pass, self.db_host))

        create_handler_context(self)

    async def stop(self):
        await self._async_exit_stack.aclose()
        self._exit_stack.close()

    @acm
    async def open(self):
        await self.start()
        yield self
        await self.stop()

    async def update_status_message(
        self, status_msg, new_text: str, **kwargs
    ):
        await self.db_call(
            'update_user_request_by_sid', status_msg.id, new_text)
        return await self.bot.edit_message_text(
            new_text,
            chat_id=status_msg.chat.id,
            message_id=status_msg.id,
            **kwargs
        )

    async def append_status_message(
        self, status_msg, add_text: str, **kwargs
    ):
        request = await self.db_call('get_user_request_by_sid', status_msg.id)
        await self.update_status_message(
            status_msg,
            request['status'] + add_text,
            **kwargs
        )

    async def work_request(
        self,
        user,
        status_msg,
        method: str,
        params: dict,
        file_id: str | None = None,
        binary_data: str = ''
    ):
        if params['seed'] == None:
            params['seed'] = random.randint(0, 0xFFFFFFFF)

        sanitized_params = {}
        for key, val in params.items():
            if isinstance(val, Decimal):
                val = str(val)

            sanitized_params[key] = val

        body = json.dumps({
            'method': 'diffuse',
            'params': sanitized_params
        })
        request_time = datetime.now().isoformat()

        await self.update_status_message(
            status_msg,
            f'processing a \'{method}\' request by {tg_user_pretty(user)}\n'
            f'[{timestamp_pretty()}] <i>broadcasting transaction to chain...</i>',
            parse_mode='HTML'
        )

        reward = '20.0000 GPU'
        res = await self.cleos.a_push_action(
            'telos.gpu',
            'enqueue',
            {
                'user': Name(self.account),
                'request_body': body,
                'binary_data': binary_data,
                'reward': asset_from_str(reward)
            },
            self.account, self.key, permission=self.permission
        )

        if 'code' in res or 'statusCode' in res:
            logging.error(json.dumps(res, indent=4))
            await self.update_status_message(
                status_msg,
                'skynet has suffered an internal error trying to fill this request')
            return

        enqueue_tx_id = res['transaction_id']
        enqueue_tx_link = hlink(
            'Your request on Skynet Explorer',
            f'https://skynet.ancap.tech/v2/explore/transaction/{enqueue_tx_id}'
        )

        await self.append_status_message(
            status_msg,
            f' <b>broadcasted!</b>\n'
            f'<b>{enqueue_tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>workers are processing request...</i>',
            parse_mode='HTML'
        )

        out = collect_stdout(res)

        request_id, nonce = out.split(':')

        request_hash = sha256(
            (nonce + body + binary_data).encode('utf-8')).hexdigest().upper()

        request_id = int(request_id)

        logging.info(f'{request_id} enqueued.')

        tx_hash = None
        ipfs_hash = None
        for i in range(60):
            submits = await self.hyperion.aget_actions(
                account=self.account,
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
            await self.update_status_message(
                status_msg,
                '\n[{timestamp_pretty()}] <b>timeout processing request</b>',
                parse_mode='HTML'
            )
            return

        tx_link = hlink(
            'Your result on Skynet Explorer',
            f'https://skynet.ancap.tech/v2/explore/transaction/{tx_hash}'
        )

        await self.append_status_message(
            status_msg,
            f' <b>request processed!</b>\n'
            f'<b>{tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>trying to download image...</i>\n',
            parse_mode='HTML'
        )

        # attempt to get the image and send it
        ipfs_link = f'https://ipfs.ancap.tech/ipfs/{ipfs_hash}/image.png'
        resp = await get_ipfs_file(ipfs_link)

        caption = generate_reply_caption(
            user, params, ipfs_hash, tx_hash, worker, reward)

        if not resp or resp.status_code != 200:
            logging.error(f'couldn\'t get ipfs hosted image at {ipfs_link}!')
            await self.update_status_message(
                status_msg,
                caption,
                reply_markup=build_redo_menu(),
                parse_mode='HTML'
            )

        else:
            logging.info(f'success! sending generated image')
            await self.bot.delete_message(
                chat_id=status_msg.chat.id, message_id=status_msg.id)
            if file_id:  # img2img
                await self.bot.send_media_group(
                    status_msg.chat.id,
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
                await self.bot.send_photo(
                    status_msg.chat.id,
                    caption=caption,
                    photo=resp.raw,
                    reply_markup=build_redo_menu(),
                    parse_mode='HTML'
                )
