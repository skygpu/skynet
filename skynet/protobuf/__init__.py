#!/usr/bin/python

from typing import Optional
from dataclasses import dataclass, asdict

from google.protobuf.json_format import MessageToDict

from .auth import *
from .skynet_pb2 import *


class Struct:

    def to_dict(self):
        return asdict(self)


@dataclass
class Text2ImageParameters(Struct):
    algo: str
    prompt: str
    step: int
    width: int
    height: int
    guidance: float
    seed: Optional[int]
    upscaler: Optional[str]
