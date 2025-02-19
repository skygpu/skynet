import logging
import traceback

from typing import Self, Awaitable
from datetime import datetime, timezone

from telebot.types import (
    User as TGUser,
    Chat as TGChat,
    PhotoSize as TGPhotoSize,
    Message as TGMessage,
    InputMediaPhoto,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telebot.async_telebot import AsyncTeleBot, ExceptionHandler
from telebot.formatting import hlink

from skynet.types import BodyV0Params
from skynet.config import FrontendConfig
from skynet.constants import VERSION
from skynet.frontend.chatbot import BaseChatbot
from skynet.frontend.chatbot.db import FrontendUserDB
from skynet.frontend.chatbot.types import (
    BaseUser,
    BaseChatRoom,
    BaseFileInput,
    BaseCommands,
    BaseMessage
)

GROUP_ID = -1001541979235
TEST_GROUP_ID = -4099622703
ADMIN_USER_ID = 383385940


# Chatbot types impls

class TelegramUser(BaseUser):
    def __init__(self, user: TGUser):
        self._user = user

    @property
    def id(self) -> int:
        return self._user.id

    @property
    def name(self) -> str:
        if self._user.username:
            return f'@{self._user.username}'

        return f'{self._user.first_name} id: {self.id}'

    @property
    def is_admin(self) -> bool:
        return self.id == ADMIN_USER_ID


class TelegramChatRoom(BaseChatRoom):

    def __init__(self, chat: TGChat):
        self._chat = chat

    @property
    def id(self) -> int:
        return self._chat.id

    @property
    def is_private(self) -> bool:
        return self._chat.type == 'private'


class TelegramFileInput(BaseFileInput):

    def __init__(
        self,
        photo: TGPhotoSize | None = None,
        id: int | None = None,
        cid: str | None = None
    ):
        self._photo = photo
        self._id = id
        self._cid = cid

        self._raw = None

    def from_values(id: int, cid: str) -> Self:
        return TelegramFileInput(id=id, cid=cid)

    @property
    def id(self) -> int:
        if self._id:
            return self._id

        return self._photo.file_id

    @property
    def cid(self) -> str:
        if self._cid:
            return self._cid

        raise ValueError

    def set_cid(self, cid: str):
        self._cid = cid

    async def download(self, bot: AsyncTeleBot) -> bytes:
        file_path = (await bot.get_file(self.id)).file_path
        self._raw = await bot.download_file(file_path)
        return self._raw


class TelegramMessage(BaseMessage):

    def __init__(self, cmd: BaseCommands | None, msg: TGMessage):
        self._msg = msg
        self._cmd = cmd
        self._chat = TelegramChatRoom(msg.chat)
        self._inputs: list[TelegramFileInput] | None = None

    @property
    def id(self) -> int:
        return self._msg.message_id

    @property
    def chat(self) -> TelegramChatRoom:
        return self._chat

    @property
    def text(self) -> str:
        # remove command name, slash and first space
        if self._msg.text:
            return self._msg.text[len(self._cmd) + 2:]

        return self._msg.caption[len(self._cmd) + 2:]

    @property
    def author(self) -> TelegramUser:
        return TelegramUser(self._msg.from_user)

    @property
    def command(self) -> str | None:
        return self._cmd

    @property
    def inputs(self) -> list[TelegramFileInput]:
        if self._inputs is None:
            self._inputs = []
            if self._msg.photo:
                self._inputs = [
                    TelegramFileInput(photo=p)
                    for p in self._msg.photo
                ]

        return self._inputs


# generic tg utils

def timestamp_pretty():
    return datetime.now(timezone.utc).strftime('%H:%M:%S')


class TGExceptionHandler(ExceptionHandler):

    def handle(exception):
        traceback.print_exc()


def build_redo_menu():
    btn_redo = InlineKeyboardButton("Redo", callback_data='{\"method\": \"redo\"}')
    inline_keyboard = InlineKeyboardMarkup()
    inline_keyboard.add(btn_redo)
    return inline_keyboard


def prepare_metainfo_caption(user: TelegramUser, worker: str, reward: str, params: BodyV0Params) -> str:
    prompt = params.prompt
    if len(prompt) > 256:
        prompt = prompt[:256]

    meta_str = f'<u>by {user.name}</u>\n'
    meta_str += f'<i>performed by {worker}</i>\n'
    meta_str += f'<b><u>reward: {reward}</u></b>\n'

    meta_str += f'<code>prompt:</code> {prompt}\n'
    meta_str += f'<code>seed: {params.seed}</code>\n'
    meta_str += f'<code>step: {params.step}</code>\n'
    if params.guidance:
        meta_str += f'<code>guidance: {params.guidance}</code>\n'

    if params.strength:
        meta_str += f'<code>strength: {params.strength}</code>\n'

    meta_str += f'<code>algo: {params.model}</code>\n'

    meta_str += f'<b><u>Made with Skynet v{VERSION}</u></b>\n'
    meta_str += '<b>JOIN THE SWARM: @skynetgpu</b>'
    return meta_str


def generate_reply_caption(
    config: FrontendConfig,
    user: TelegramUser,
    params: BodyV0Params,
    tx_hash: str,
    worker: str,
):
    explorer_link = hlink(
        'SKYNET Transaction Explorer',
        f'https://{config.explorer_domain}/v2/explore/transaction/{tx_hash}'
    )

    meta_info = prepare_metainfo_caption(user, worker, config.reward, params)

    final_msg = '\n'.join([
        'Worker finished your task!',
        explorer_link,
        f'PARAMETER INFO:\n{meta_info}'
    ])

    final_msg = '\n'.join([
        f'<b><i>{explorer_link}</i></b>',
        f'{meta_info}'
    ])

    return final_msg


def append_handler(bot: AsyncTeleBot, command: str, fn: Awaitable):
    @bot.message_handler(commands=[command])
    async def wrap_msg_and_handle(tg_msg: TGMessage):
        await fn(TelegramMessage(cmd=command, msg=tg_msg))


class TelegramChatbot(BaseChatbot):

    def __init__(
        self,
        config: FrontendConfig,
        db: FrontendUserDB,
    ):
        super().__init__(config, db)
        bot = AsyncTeleBot(config.token, exception_handler=TGExceptionHandler)

        append_handler(bot, BaseCommands.HELP, self.send_help)
        append_handler(bot, BaseCommands.COOL, self.send_cool_words)
        append_handler(bot, BaseCommands.QUEUE, self.get_queue)
        append_handler(bot, BaseCommands.CONFIG, self.set_config)
        append_handler(bot, BaseCommands.STATS, self.user_stats)
        append_handler(bot, BaseCommands.DONATE, self.donation_info)
        append_handler(bot, BaseCommands.SAY, self.say)

        append_handler(bot, BaseCommands.TXT2IMG, self.handle_request)

        append_handler(bot, BaseCommands.IMG2IMG, self.handle_request)

        @bot.message_handler(func=lambda _: True, content_types=['photo', 'document'])
        async def handle_img2img(tg_msg: TGMessage):
            msg = TelegramMessage(cmd='img2img', msg=tg_msg)
            for file in msg.inputs:
                await file.download(bot)
            await self.handle_request(msg)

        append_handler(bot, BaseCommands.REDO, self.handle_request)

        self.bot = bot

        self._main_room: TelegramChatRoom | None = None

    async def init(self):
        tg_group = await self.bot.get_chat(TEST_GROUP_ID)
        self._main_room = TelegramChatRoom(chat=tg_group)
        logging.info('initialized')

    async def run(self):
        await self.init()
        await self.bot.infinity_polling()

    @property
    def main_group(self) -> TelegramChatRoom:
        return self._main_room

    async def new_msg(self, chat: TelegramChatRoom, text: str) -> TelegramMessage:
        msg = await self.bot.send_message(chat.id, text, parse_mode='HTML')
        return TelegramMessage(cmd=None, msg=msg)

    async def reply_to(self, msg: TelegramMessage, text: str) -> TelegramMessage:
        msg = await self.bot.reply_to(msg._msg, text, parse_mode='HTML')
        return TelegramMessage(cmd=None, msg=msg)

    async def edit_msg(self, msg: TelegramMessage, text: str):
        await self.bot.edit_message_text(
            text,
            chat_id=msg.chat.id,
            message_id=msg.id,
            parse_mode='HTML'
        )

    async def update_request_status_timeout(self, status_msg: TelegramMessage):
        '''
        Notify users when we timedout trying to find a matching submit
        '''
        await self.append_status_msg(
            status_msg,
            f'\n[{timestamp_pretty()}] <b>timeout processing request</b>',
        )

    async def update_request_status_step_0(self, status_msg: TelegramMessage, user_msg: TelegramMessage):
        '''
        First step in request status message lifecycle, should notify which user sent the request
        and that we are about to broadcast the request to chain
        '''
        await self.update_status_msg(
            status_msg,
            f'processing a \'{user_msg.command}\' request by {user_msg.author.name}\n'
            f'[{timestamp_pretty()}] <i>broadcasting transaction to chain...</i>'
        )

    async def update_request_status_step_1(self, status_msg: TelegramMessage, tx_result: dict):
        '''
        Second step in request status message lifecycle, should notify enqueue transaction
        was processed by chain, and provide a link to the tx in the chain explorer
        '''
        enqueue_tx_id = tx_result['transaction_id']
        enqueue_tx_link = hlink(
            'Your request on Skynet Explorer',
            f'https://{self.config.explorer_domain}/v2/explore/transaction/{enqueue_tx_id}'
        )
        await self.append_status_msg(
            status_msg,
            f' <b>broadcasted!</b>\n'
            f'<b>{enqueue_tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>workers are processing request...</i>',
        )

    async def update_request_status_step_2(self, status_msg: TelegramMessage, submit_tx_hash: str):
        '''
        Third step in request status message lifecycle, should notify matching submit transaction
        was found, and provide a link to the tx in the chain explorer
        '''
        tx_link = hlink(
            'Your result on Skynet Explorer',
            f'https://{self.config.explorer_domain}/v2/explore/transaction/{submit_tx_hash}'
        )
        await self.append_status_msg(
            status_msg,
            f' <b>request processed!</b>\n'
            f'<b>{tx_link}</b>\n'
            f'[{timestamp_pretty()}] <i>trying to download image...</i>\n',
        )

    async def update_request_status_final(
        self,
        og_msg: TelegramMessage,
        status_msg: TelegramMessage,
        user: TelegramUser,
        params: BodyV0Params,
        inputs: list[TelegramFileInput],
        submit_tx_hash: str,
        worker: str,
        result_img: bytes | None
    ):
        '''
        Last step in request status message lifecycle, should delete status message and send a
        new message replying to the original user's message, generate the appropiate
        reply caption and if provided also sent the found result img
        '''
        caption = generate_reply_caption(
            self.config, user, params, submit_tx_hash, worker)

        await self.bot.delete_message(
            chat_id=status_msg.chat.id,
            message_id=status_msg.id
        )

        if not result_img:
            # result found on chain but failed to fetch img from ipfs
            await self.reply_to(og_msg, caption, reply_markup=build_redo_menu())
            return

        match len(inputs):
            case 0:
                await self.bot.send_photo(
                    status_msg.chat.id,
                    caption=caption,
                    photo=result_img,
                    reply_markup=build_redo_menu(),
                    parse_mode='HTML'
                )

            case _:
                _input = inputs[-1]
                await self.bot.send_media_group(
                    status_msg.chat.id,
                    media=[
                        InputMediaPhoto(_input.id),
                        InputMediaPhoto(result_img, caption=caption, parse_mode='HTML')
                    ]
                )
