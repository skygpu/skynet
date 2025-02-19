import io
import json
import asyncio
import logging
from abc import ABC, abstractmethod, abstractproperty
from PIL import Image, UnidentifiedImageError
from random import randint
from decimal import Decimal
from hashlib import sha256
from datetime import datetime, timedelta

import msgspec
from leap import CLEOS
from leap.hyperion import HyperionAPI

from skynet.ipfs import AsyncIPFSHTTP, get_ipfs_file
from skynet.types import BodyV0, BodyV0Params
from skynet.config import FrontendConfig
from skynet.constants import (
    MODELS, GPU_CONTRACT_ABI,
    HELP_TEXT,
    HELP_TOPICS,
    HELP_UNKWNOWN_PARAM,
    COOL_WORDS,
    DONATION_INFO
)
from skynet.frontend import validate_user_config_request
from skynet.frontend.chatbot.db import FrontendUserDB
from skynet.frontend.chatbot.types import (
    BaseUser,
    BaseChatRoom,
    BaseCommands,
    BaseFileInput,
    BaseMessage
)


def perform_auto_conf(config: dict) -> dict:
    model = MODELS[config['model']]

    maybe_step = model.attrs.get('step', None)
    if maybe_step:
        config['step'] = maybe_step

    maybe_width = model.attrs.get('width', None)
    if maybe_width:
        config['width'] = maybe_step

    maybe_height = model.attrs.get('height', None)
    if maybe_height:
        config['height'] = maybe_step

    return config


def sanitize_params(params: dict) -> dict:
    if (
        'seed' not in params
        or
        params['seed'] is None
    ):
        params['seed'] = randint(0, 0xffffffff)

    s_params = {}
    for key, val in params.items():
        if isinstance(val, Decimal):
            val = str(val)

        s_params[key] = val

    return s_params


class RequestTimeoutError(BaseException):
    ...


class BaseChatbot(ABC):

    def __init__(
        self,
        config: FrontendConfig,
        db: FrontendUserDB
    ):
        self.db = db
        self.config = config
        self.ipfs = AsyncIPFSHTTP(config.ipfs_url)
        self.cleos = CLEOS(endpoint=config.node_url)
        self.cleos.load_abi(config.receiver, GPU_CONTRACT_ABI)
        self.cleos.import_key(config.account, config.key)
        self.hyperion = HyperionAPI(config.hyperion_url)

    async def init(self):
        ...

    @abstractmethod
    async def run(self):
        ...

    @abstractproperty
    def main_group(self) -> BaseChatRoom:
        ...

    @abstractmethod
    async def new_msg(self, chat: BaseChatRoom, text: str, **kwargs) -> BaseMessage:
        '''
        Send text to a chat/channel.
        '''
        ...

    @abstractmethod
    async def reply_to(self, msg: BaseMessage, text: str, **kwargs) -> BaseMessage:
        '''
        Reply to existing message by sending new message.
        '''
        ...

    @abstractmethod
    async def edit_msg(self, msg: BaseMessage, text: str, **kwargs):
        '''
        Edit an existing message.
        '''
        ...

    async def create_status_msg(self, msg: BaseMessage, init_text: str) -> tuple[BaseUser, BaseMessage, dict]:
        # maybe init user
        user = msg.author
        user_row = await self.db.get_or_create_user(user.id)

        # create status msg
        status_msg = await self.reply_to(msg, init_text)

        # start tracking of request in db
        await self.db.new_user_request(user.id, msg.id, status_msg.id, status=init_text)
        return [user, status_msg, user_row]

    async def update_status_msg(self, msg: BaseMessage, text: str):
        '''
        Update an existing status message, also mirrors changes on db
        '''
        await self.db.update_user_request_by_sid(msg.id, text)
        await self.edit_msg(msg, text)

    async def append_status_msg(self, msg: BaseMessage, text: str):
        '''
        Append text to an existing status message
        '''
        request = await self.db.get_user_request_by_sid(msg.id)
        await self.update_status_msg(msg, request['status'] + text)

    @abstractmethod
    async def update_request_status_timeout(self, status_msg: BaseMessage):
        '''
        Notify users when we timedout trying to find a matching submit
        '''
        ...

    @abstractmethod
    async def update_request_status_step_0(self, status_msg: BaseMessage, user_msg: BaseMessage):
        '''
        First step in request status message lifecycle, should notify which user sent the request
        and that we are about to broadcast the request to chain
        '''
        ...

    @abstractmethod
    async def update_request_status_step_1(self, status_msg: BaseMessage, tx_result: dict):
        '''
        Second step in request status message lifecycle, should notify enqueue transaction
        was processed by chain, and provide a link to the tx in the chain explorer
        '''
        ...

    @abstractmethod
    async def update_request_status_step_2(self, status_msg: BaseMessage, submit_tx_hash: str):
        '''
        Third step in request status message lifecycle, should notify matching submit transaction
        was found, and provide a link to the tx in the chain explorer
        '''
        ...

    @abstractmethod
    async def update_request_status_final(
        self,
        og_msg: BaseMessage,
        status_msg: BaseMessage,
        user: BaseUser,
        params: BodyV0Params,
        inputs: list[BaseFileInput],
        submit_tx_hash: str,
        worker: str,
        result_img: bytes | None
    ):
        '''
        Last step in request status message lifecycle, should delete status message and send a
        new message replying to the original user's message, generate the appropiate
        reply caption and if provided also sent the found result img
        '''
        ...


    async def handle_request(
        self,
        msg: BaseMessage
    ):
        if msg.chat.is_private:
            return

        if (
            len(msg.text) == 0
            and
            msg.command != BaseCommands.REDO
        ):
            await self.reply_to(msg, 'empty prompt ignored.')
            return

        # maybe initialize user db row and send a new msg thats gonna
        # be updated throughout the request lifecycle
        user, status_msg, user_row = await self.create_status_msg(
            msg, f'started processing a {msg.command} request...')

        # if this is a redo msg, we attempt to get the input params from db
        # else use msg properties
        match msg.command:
            case BaseCommands.TXT2IMG | BaseCommands.IMG2IMG:
                prompt = msg.text
                command = msg.command
                inputs = msg.inputs

            case BaseCommands.REDO:
                prompt = await self.db.get_last_prompt_of(user.id)
                command = await self.db.get_last_method_of(user.id)
                inputs = await self.db.get_last_inputs_of(user.id)

                if not prompt:
                    await self.reply_to(msg, 'no last prompt found, try doing a non-redo request first')
                    return

            case _:
                await self.reply_to(msg, f'unknown request of type {msg.command}')
                return

        if (
            msg.command == BaseCommands.IMG2IMG
            and
            len(inputs) == 0
        ):
            await self.edit_msg(status_msg, 'seems you tried to do an img2img command without sending image')
            return

        # maybe apply recomended settings to this request
        del user_row['id']
        if user_row['autoconf']:
            user_row = perform_auto_conf(user_row)

        user_row = sanitize_params(user_row)

        body = BodyV0(
            method=command,
            params=BodyV0Params(
                prompt=prompt,
                **user_row
            )
        )

        # publish inputs to ipfs
        input_cids = []
        for i in inputs:
            await i.publish(self.ipfs, user_row)
            input_cids.append(i.cid)

        inputs_str = ','.join((i for i in input_cids))

        # unless its a redo request, update db user data
        if command != BaseCommands.REDO:
            await self.db.update_user_stats(
                user.id,
                command,
                last_prompt=prompt,
                last_inputs=inputs
            )

        await self.update_request_status_step_0(status_msg, msg)

        # prepare and send enqueue request
        request_time = datetime.now().isoformat()
        str_body = msgspec.json.encode(body).decode('utf-8')

        enqueue_receipt = await self.cleos.a_push_action(
            self.config.receiver,
            'enqueue',
            [
                self.config.account,
                str_body,
                inputs_str,
                self.config.reward,
                1
            ],
            self.config.account,
            key=self.cleos.private_keys[self.config.account],
            permission=self.config.permission
        )

        await self.update_request_status_step_1(status_msg, enqueue_receipt)

        # wait and search submit request using hyperion endpoint
        console = enqueue_receipt['processed']['action_traces'][0]['console']
        console_lines = console.split('\n')

        request_id = None
        request_hash = None
        if self.config.proto_version == 0:
            '''
            v0 has req_id:nonce printed in enqueue console output
            to search for a result request_hash arg on submit has
            to match the sha256 of nonce + body + input_str
            '''
            request_id, nonce = console_lines[-1].rstrip().split(':')
            request_hash = sha256(
                (nonce + str_body + inputs_str).encode('utf-8')).hexdigest().upper()

            request_id = int(request_id)

        elif self.config.proto_version == 1:
            '''
            v1 uses a global unique nonce and prints it on enqueue
            console output to search for a result request_id arg
            on submit has to match the printed req_id
            '''
            request_id = int(console_lines[-1].rstrip())

        else:
            raise NotImplementedError

        worker = None
        submit_tx_hash = None
        result_cid = None
        for i in range(1, self.config.request_timeout + 1):
            try:
                submits = await self.hyperion.aget_actions(
                    account=self.config.account,
                    filter=f'{self.config.receiver}:submit',
                    sort='desc',
                    after=request_time
                )
                if self.config.proto_version == 0:
                    actions = [
                        action
                        for action in submits['actions']
                        if action['act']['data']['request_hash'] == request_hash
                    ]
                elif self.config.proto_version == 1:
                    actions = [
                        action
                        for action in submits['actions']
                        if action['act']['data']['request_id'] == request_id
                    ]

                else:
                    raise NotImplementedError

                if len(actions) > 0:
                    action = actions[0]
                    submit_tx_hash = action['trx_id']
                    data = action['act']['data']
                    result_cid = data['ipfs_hash']
                    worker = data['worker']
                    logging.info(f'found matching submit! tx: {submit_tx_hash} cid: {result_cid}')
                    break

            except json.JSONDecodeError:
                if i < self.config.request_timeout:
                    logging.error('network error while searching for submit, retry...')

            await asyncio.sleep(1)

        # if we found matching submit submit_tx_hash, worker, and result_cid will not be None
        if not result_cid:
            await self.update_request_status_timeout(status_msg)
            raise RequestTimeoutError

        await self.update_request_status_step_2(status_msg, submit_tx_hash)

        # attempt to get the image and send it
        result_link = f'https://{self.config.ipfs_domain}/ipfs/{result_cid}'
        get_img_response = await get_ipfs_file(result_link)

        result_img = None
        if get_img_response and get_img_response.status_code == 200:
            try:
                with Image.open(io.BytesIO(get_img_response.read())) as img:
                    w, h = img.size

                    if (
                        w > self.config.result_max_width
                        or
                        h > self.config.result_max_height
                    ):
                        max_size = (self.config.result_max_width, self.config.result_max_height)
                        logging.warning(
                            f'raw result is of size {img.size}, resizing to {max_size}')
                        img.thumbnail(max_size)

                    tmp_buf = io.BytesIO()
                    img.save(tmp_buf, format='PNG')
                    result_img = tmp_buf.getvalue()

            except UnidentifiedImageError:
                logging.warning(f'couldn\'t get ipfs result at {result_link}!')

        await self.update_request_status_final(
            msg, status_msg, user, body.params, inputs, submit_tx_hash, worker, result_img)

        await self.db.increment_generated(user.id)

    async def send_help(self, msg: BaseMessage):
        if len(msg.text) == 0:
            await self.reply_to(msg, HELP_TEXT)

        else:
            if msg.text in HELP_TOPICS:
                await self.reply_to(msg, HELP_TOPICS[msg.text])

            else:
                await self.reply_to(msg, HELP_UNKWNOWN_PARAM)

    async def send_cool_words(self, msg: BaseMessage):
        await self.reply_to(msg, '\n'.join(COOL_WORDS))

    async def get_queue(self, msg: BaseMessage):
        an_hour_ago = datetime.now() - timedelta(hours=1)
        queue = await self.cleos.aget_table(
            self.config.receiver, self.config.receiver, 'queue',
            index_position=2,
            key_type='i64',
            sort='desc',
            lower_bound=int(an_hour_ago.timestamp())
        )
        await self.reply_to(
            msg, f'Requests on skynet queue: {len(queue)}')

    async def set_config(self, msg: BaseMessage):
        try:
            attr, val, reply_txt = validate_user_config_request(msg.text)

            await self.db.update_user_config(msg.author.id, attr, val)

        except BaseException as e:
            reply_txt = str(e)

        finally:
            await self.reply_to(msg, reply_txt)

    async def user_stats(self, msg: BaseMessage):
        await self.db.get_or_create_user(msg.author.id)
        generated, joined, role = await self.db.get_user_stats(msg.author.id)

        stats_str = f'generated: {generated}\n'
        stats_str += f'joined: {joined}\n'
        stats_str += f'role: {role}\n'

        await self.reply_to(msg, stats_str)

    async def donation_info(self, msg: BaseMessage):
        await self.reply_to(msg, DONATION_INFO)

    async def say(self, msg: BaseMessage):
        if (
            msg.chat.is_private
            or
            not msg.author.is_admin
        ):
            return

        await self.new_msg(self.main_group, msg.text)
