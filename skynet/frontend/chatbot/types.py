import io

from abc import ABC, abstractproperty, abstractmethod
from enum import StrEnum
from typing import Self
from PIL import Image

from skynet.ipfs import AsyncIPFSHTTP


class BaseUser(ABC):

    @abstractproperty
    def id(self) -> int:
        ...

    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def is_admin(self) -> bool:
        ...


class BaseChatRoom(ABC):
    @abstractproperty
    def id(self) -> int:
        ...

    @abstractproperty
    def is_private(self) -> bool:
        ...


class BaseFileInput(ABC):

    @staticmethod
    @abstractmethod
    def from_values(id: int, cid: str) -> Self:
        ...

    @abstractproperty
    def id(self) -> int:
        ...

    @abstractproperty
    def cid(self) -> str:
        ...

    @abstractmethod
    async def download(self, *args) -> bytes:
        ...

    async def publish(self, ipfs_api: AsyncIPFSHTTP, user_row: dict):
        with Image.open(io.BytesIO(self._raw)) as img:
            w, h = img.size

            if (
                w > user_row['width']
                or
                h > user_row['height']
            ):
                img.thumbnail((user_row['width'], user_row['height']))

            img_path = '/tmp/ipfs-staging/img.png'
            img.save(img_path, format='PNG')

            ipfs_info = await ipfs_api.add(img_path)
            ipfs_hash = ipfs_info['Hash']
            await ipfs_api.pin(ipfs_hash)


class BaseCommands(StrEnum):
    TXT2IMG = 'txt2img'
    IMG2IMG = 'img2img'
    REDO    = 'redo'
    HELP    = 'help'
    COOL    = 'cool'
    QUEUE   = 'queue'
    CONFIG  = 'config'
    STATS   = 'stats'
    DONATE  = 'donate'
    SAY     = 'say'


class BaseMessage(ABC):
    @abstractproperty
    def id(self) -> int:
        ...

    @abstractproperty
    def chat(self) -> BaseChatRoom:
        ...

    @abstractproperty
    def text(self) -> str:
        ...

    @abstractproperty
    def author(self) -> BaseUser:
        ...

    @abstractproperty
    def command(self) -> str | None:
        ...

    @abstractproperty
    def inputs(self) -> list[BaseFileInput]:
        ...
