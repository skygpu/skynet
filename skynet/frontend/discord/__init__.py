#!/usr/bin/python

from json import JSONDecodeError
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
# from telebot.types import InputMediaPhoto

import discord
import io
from PIL import Image, UnidentifiedImageError

from skynet.db import open_database_connection
from skynet.ipfs import get_ipfs_file, AsyncIPFSHTTP
from skynet.constants import *

from . import *
from .bot import DiscordBot

from .utils import *
from .handlers import create_handler_context
from .ui import SkynetView


class SkynetDiscordFrontend:

    def __init__(
        self,
        # token: str,
        account: str,
        permission: str,
        node_url: str,
        hyperion_url: str,
        db_host: str,
        db_user: str,
        db_pass: str,
        ipfs_url: str,
        remote_ipfs_node: str,
        key: str,
        explorer_domain: str,
        ipfs_domain: str
    ):
        # self.token = token
        self.account = account
        self.permission = permission
        self.node_url = node_url
        self.hyperion_url = hyperion_url
        self.db_host = db_host
        self.db_user = db_user
        self.db_pass = db_pass
        self.ipfs_url = ipfs_url
        self.remote_ipfs_node = remote_ipfs_node
        self.key = key
        self.explorer_domain = explorer_domain
        self.ipfs_domain = ipfs_domain

        self.bot = DiscordBot(self)
        self.cleos = CLEOS(None, None, url=node_url, remote=node_url)
        self.hyperion = HyperionAPI(hyperion_url)
        self.ipfs_node = AsyncIPFSHTTP(ipfs_url)

        self._exit_stack = ExitStack()
        self._async_exit_stack = AsyncExitStack()

    async def start(self):
        if self.remote_ipfs_node:
            await self.ipfs_node.connect(self.remote_ipfs_node)

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

    # maybe do this?
    # async def update_status_message(
    #     self, status_msg, new_text: str, **kwargs
    # ):
    #     await self.db_call(
    #         'update_user_request_by_sid', status_msg.id, new_text)
    #     return await self.bot.edit_message_text(
    #         new_text,
    #         chat_id=status_msg.chat.id,
    #         message_id=status_msg.id,
    #         **kwargs
    #     )

    # async def append_status_message(
    #     self, status_msg, add_text: str, **kwargs
    # ):
    #     request = await self.db_call('get_user_request_by_sid', status_msg.id)
    #     await self.update_status_message(
    #         status_msg,
    #         request['status'] + add_text,
    #         **kwargs
    #     )

    async def work_request(
        self,
        user,
        status_msg,
        method: str,
        params: dict,
        ctx: discord.ext.commands.context.Context | discord.Message,
        file_id: str | None = None,
        binary_data: str = ''
    ) -> bool:
        send = ctx.channel.send

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

        await status_msg.delete()
        msg_text = f'processing a \'{method}\' request by {user.name}\n[{timestamp_pretty()}] *broadcasting transaction to chain...* '
        embed = discord.Embed(
            title='live updates',
            description=msg_text,
            color=discord.Color.blue())

        message = await send(embed=embed)

        reward = '20.0000 GPU'
        res = await self.cleos.a_push_action(
            'telos.gpu',
            'enqueue',
            {
                'user': Name(self.account),
                'request_body': body,
                'binary_data': binary_data,
                'reward': asset_from_str(reward),
                'min_verification': 1
            },
            self.account, self.key, permission=self.permission
        )

        if 'code' in res or 'statusCode' in res:
            logging.error(json.dumps(res, indent=4))
            await self.bot.channel.send(
                status_msg,
                'skynet has suffered an internal error trying to fill this request')
            return False

        enqueue_tx_id = res['transaction_id']
        enqueue_tx_link = f'[**Your request on Skynet Explorer**](https://{self.explorer_domain}/v2/explore/transaction/{enqueue_tx_id})'

        msg_text += f'**broadcasted!** \n{enqueue_tx_link}\n[{timestamp_pretty()}] *workers are processing request...* '
        embed = discord.Embed(
            title='live updates',
            description=msg_text,
            color=discord.Color.blue())

        await message.edit(embed=embed)

        out = collect_stdout(res)

        request_id, nonce = out.split(':')

        request_hash = sha256(
            (nonce + body + binary_data).encode('utf-8')).hexdigest().upper()

        request_id = int(request_id)

        logging.info(f'{request_id} enqueued.')

        tx_hash = None
        ipfs_hash = None
        for i in range(60):
            try:
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

            except JSONDecodeError:
                logging.error(f'network error while getting actions, retry..')

            await asyncio.sleep(1)

        if not ipfs_hash:

            timeout_text = f'\n[{timestamp_pretty()}] **timeout processing request**'
            embed = discord.Embed(
                title='live updates',
                description=timeout_text,
                color=discord.Color.blue())

            await message.edit(embed=embed)
            return False

        tx_link = f'[**Your result on Skynet Explorer**](https://{self.explorer_domain}/v2/explore/transaction/{tx_hash})'

        msg_text += f'**request processed!**\n{tx_link}\n[{timestamp_pretty()}] *trying to download image...*\n '
        embed = discord.Embed(
            title='live updates',
            description=msg_text,
            color=discord.Color.blue())

        await message.edit(embed=embed)

        # attempt to get the image and send it
        results = {}
        ipfs_link = f'https://{self.ipfs_domain}/ipfs/{ipfs_hash}'
        ipfs_link_legacy = ipfs_link + '/image.png'

        async def get_and_set_results(link: str):
            res = await get_ipfs_file(link)
            logging.info(f'got response from {link}')
            if not res or res.status_code != 200:
                logging.warning(f'couldn\'t get ipfs binary data at {link}!')

            else:
                try:
                    with Image.open(io.BytesIO(res.raw)) as image:
                        tmp_buf = io.BytesIO()
                        image.save(tmp_buf, format='PNG')
                        png_img = tmp_buf.getvalue()
                        results[link] = png_img

                except UnidentifiedImageError:
                    logging.warning(
                        f'couldn\'t get ipfs binary data at {link}!')

        tasks = [
            get_and_set_results(ipfs_link),
            get_and_set_results(ipfs_link_legacy)
        ]
        await asyncio.gather(*tasks)

        png_img = None
        if ipfs_link_legacy in results:
            png_img = results[ipfs_link_legacy]

        if ipfs_link in results:
            png_img = results[ipfs_link]

        if not png_img:
            logging.error(f'couldn\'t get ipfs hosted image at {ipfs_link}!')
            embed.add_field(
                name='Error', value=f'couldn\'t get ipfs hosted image [**here**]({ipfs_link})!')
            await message.edit(embed=embed, view=SkynetView(self))
            return True

        # reword this function, may not need caption
        caption, embed = generate_reply_caption(
            user, params, tx_hash, worker, reward, self.explorer_domain)

        logging.info(f'success! sending generated image')
        await message.delete()
        if file_id:  # img2img
            embed.set_thumbnail(
                url='https://ipfs.skygpu.net/ipfs/' + binary_data + '/image.png')
            embed.set_image(url=ipfs_link)
            await send(embed=embed, view=SkynetView(self))
        else:  # txt2img
            embed.set_image(url=ipfs_link)
            await send(embed=embed, view=SkynetView(self))

        return True
