import io
import random
import logging
import asyncio

import time
import httpx

from PIL import Image, UnidentifiedImageError
from json import JSONDecodeError
from decimal import Decimal
from hashlib import sha256
from datetime import datetime
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager as acm

from leap.cleos import CLEOS
from leap.protocol import Name, Asset
# from leap.hyperion import HyperionAPI

from telebot.types import InputMediaPhoto
from telebot.async_telebot import AsyncTeleBot

from skynet.db import open_database_connection
from skynet.ipfs import get_ipfs_file, AsyncIPFSHTTP
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
        ipfs_node: str,
        key: str,
        explorer_domain: str,
        ipfs_domain: str
    ):
        self.token = token
        self.account = account
        self.permission = permission
        self.node_url = node_url
        self.hyperion_url = hyperion_url
        self.db_host = db_host
        self.db_user = db_user
        self.db_pass = db_pass
        self.key = key
        self.explorer_domain = explorer_domain
        self.ipfs_domain = ipfs_domain

        self.bot = AsyncTeleBot(token, exception_handler=SKYExceptionHandler)
        self.cleos = CLEOS(endpoint=node_url)
        self.cleos.load_abi('gpu.scd', GPU_CONTRACT_ABI)
        # self.hyperion = HyperionAPI(hyperion_url)
        self.ipfs_node = AsyncIPFSHTTP(ipfs_node)

        self._async_exit_stack = AsyncExitStack()

    async def start(self):
        self.db_call = await self._async_exit_stack.enter_async_context(
            open_database_connection(
                self.db_user, self.db_pass, self.db_host))

        create_handler_context(self)

    async def stop(self):
        await self._async_exit_stack.aclose()

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

    async def _wait_for_submit_in_blocks(
        self,
        request_hash: str,
        start_block: int,
        timeout_seconds: int = 60 * 3,
    ):
        """
        Poll /v1/chain/get_block from start_block upwards until we see
        gpu.scd::submit with the given request_hash, or we hit timeout.

        Returns (tx_id, ipfs_hash, worker) or (None, None, None) on timeout.
        """
        endpoint = self.node_url.rstrip('/')

        next_block = start_block             # inclusive
        deadline = time.monotonic() + timeout_seconds

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                # Get current head block
                info = (await client.post(
                    f'{endpoint}/v1/chain/get_info',
                    json={}
                )).json()
                head = info['head_block_num']

                # No new blocks yet, wait a bit
                if next_block > head:
                    await asyncio.sleep(0.5)
                    continue

                # Walk all blocks we haven't seen yet
                while next_block <= head:
                    try:
                        block = (await client.post(
                            f'{endpoint}/v1/chain/get_block',
                            json={'block_num_or_id': next_block}
                        )).json()
                    except (httpx.RequestError, ValueError):
                        logging.warning(f'failed to get block {next_block}, retrying...')
                        break  # leave inner loop, re-fetch head

                    for tx in block.get('transactions', []):
                        trx = tx.get('trx')
                        # Sometimes trx can be just a string (id) — skip those.
                        if isinstance(trx, str):
                            continue

                        tx_id = trx.get('id')
                        tx_obj = trx.get('transaction') or {}
                        actions = tx_obj.get('actions', []) or []

                        for act in actions:
                            if (
                                act.get('account') == 'gpu.scd'
                                and act.get('name') == 'submit'
                            ):
                                data = act.get('data') or {}
                                if data.get('request_hash') == request_hash:
                                    ipfs_hash = data.get('ipfs_hash')
                                    worker = data.get('worker')
                                    logging.info(
                                        f'Found matching submit in block {next_block}, '
                                        f'tx {tx_id}'
                                    )
                                    return tx_id, ipfs_hash, worker

                    next_block += 1

                # Caught up with head and still nothing; wait for more blocks
                await asyncio.sleep(0.5)

        return None, None, None

    async def work_request(
        self,
        user,
        status_msg,
        method: str,
        params: dict,
        file_id: str | None = None,
        inputs: list[str] = []
    ) -> bool:
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
            'gpu.scd',
            'enqueue',
            list({
                'user': Name(self.account),
                'request_body': body,
                'binary_data': ','.join(inputs),
                'reward': Asset.from_str(reward),
                'min_verification': 1
            }.values()),
            self.account, self.key, permission=self.permission
        )

        if 'code' in res or 'statusCode' in res:
            logging.error(json.dumps(res, indent=4))
            await self.update_status_message(
                status_msg,
                'skynet has suffered an internal error trying to fill this request')
            return False

        enqueue_tx_id = res['transaction_id']
        enqueue_tx_link = hlink(
            'Your request on Skynet Explorer',
            f'https://{self.explorer_domain}/v2/explore/transaction/{enqueue_tx_id}'
        )

        await self.append_status_message(
            status_msg,
            f' <b>broadcasted!</b>\n'
            f'<b>{enqueue_tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>workers are processing request...</i>',
            parse_mode='HTML'
        )

        
        out = res['processed']['action_traces'][0]['console']

        request_id, nonce = out.split(':')

        request_hash = sha256(
            (nonce + body + ','.join(inputs)).encode('utf-8')
        ).hexdigest().upper()

        request_id = int(request_id)

        logging.info(f'{request_id} enqueued.')

        # Prefer the block number from the push_transaction response
        enqueue_block_num = res.get('processed', {}).get('block_num')
        if not enqueue_block_num:
            # Fallback: start from current head if block_num is missing
            async with httpx.AsyncClient() as client:
                info = (await client.post(
                    f'{self.node_url.rstrip("/")}/v1/chain/get_info',
                    json={}
                )).json()
            enqueue_block_num = info['head_block_num']

        # Wait for submit via block polling
        tx_hash, ipfs_hash, worker = await self._wait_for_submit_in_blocks(
            request_hash=request_hash,
            start_block=enqueue_block_num,
            timeout_seconds=60 * 3,
        )

        if not ipfs_hash:
            await self.update_status_message(
                status_msg,
                f'\n[{timestamp_pretty()}] <b>timeout processing request</b>',
                parse_mode='HTML'
            )
            return False


        tx_link = hlink(
            'Your result on Skynet Explorer',
            f'https://{self.explorer_domain}/v2/explore/transaction/{tx_hash}'
        )

        await self.append_status_message(
            status_msg,
            f' <b>request processed!</b>\n'
            f'<b>{tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>trying to download image...</i>\n',
            parse_mode='HTML'
        )

        caption = generate_reply_caption(
            user, params, tx_hash, worker, reward, self.explorer_domain)

        # attempt to get the image and send it
        ipfs_link = f'https://{self.ipfs_domain}/ipfs/{ipfs_hash}'

        res = await get_ipfs_file(ipfs_link)
        logging.info(f'got response from {ipfs_link}')
        if not res or res.status_code != 200:
            logging.warning(f'couldn\'t get ipfs binary data at {ipfs_link}!')

        else:
            try:
                with Image.open(io.BytesIO(res.raw)) as image:
                    w, h = image.size

                    if w > TG_MAX_WIDTH or h > TG_MAX_HEIGHT:
                        logging.warning(f'result is of size {image.size}')
                        image.thumbnail((TG_MAX_WIDTH, TG_MAX_HEIGHT))

                    tmp_buf = io.BytesIO()
                    image.save(tmp_buf, format='PNG')
                    png_img = tmp_buf.getvalue()

            except UnidentifiedImageError:
                logging.warning(f'couldn\'t get ipfs binary data at {ipfs_link}!')

        if not png_img:
            await self.update_status_message(
                status_msg,
                caption,
                reply_markup=build_redo_menu(),
                parse_mode='HTML'
            )
            return True

        logging.info(f'success! sending generated image')
        await self.bot.delete_message(
            chat_id=status_msg.chat.id, message_id=status_msg.id)
        if file_id:  # img2img
            await self.bot.send_media_group(
                status_msg.chat.id,
                media=[
                    InputMediaPhoto(file_id),
                    InputMediaPhoto(
                        png_img,
                        caption=caption,
                        parse_mode='HTML'
                    )
                ],
            )

        else:  # txt2img
            await self.bot.send_photo(
                status_msg.chat.id,
                caption=caption,
                photo=png_img,
                reply_markup=build_redo_menu(),
                parse_mode='HTML'
            )

        return True
