from enum import StrEnum

from msgspec import Struct


class ModelMode(StrEnum):
    DIFFUSE = 'diffuse'
    TXT2IMG = 'txt2img'
    IMG2IMG = 'img2img'
    UPSCALE = 'upscale'
    INPAINT = 'inpaint'


class ModelDesc(Struct):
    short: str  # short unique name
    mem: float  # recomended mem
    attrs: dict  # additional mode specific attrs
    tags: list[ModelMode]


class BodyV0Params(Struct):
    prompt: str
    model: str
    seed: int
    step: int = 1
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    strength: float | None = None
    output_type: str | None = 'png'
    upscaler: str | None = None


class BodyV0(Struct):
    method: ModelMode
    params: BodyV0Params


class RequestV0(Struct):
    id: int
    user: str
    reward: str
    min_verification: int
    nonce: int
    body: str
    binary_data: str
    timestamp: str
