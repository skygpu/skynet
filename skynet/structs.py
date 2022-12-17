# piker: trading gear for hackers
# Copyright (C) Guillermo Rodriguez (in stewardship for piker0)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Built-in (extension) types.
"""
import sys
from typing import Optional, Union
from pprint import pformat

import msgspec


class Struct(
    msgspec.Struct,

    # https://jcristharif.com/msgspec/structs.html#tagged-unions
    # tag='pikerstruct',
    # tag=True,
):
    '''
    A "human friendlier" (aka repl buddy) struct subtype.
    '''
    def to_dict(self) -> dict:
        return {
            f: getattr(self, f)
            for f in self.__struct_fields__
        }

    def __repr__(self):
        # only turn on pprint when we detect a python REPL
        # at runtime B)
        if (
            hasattr(sys, 'ps1')
            # TODO: check if we're in pdb
        ):
            return self.pformat()

        return super().__repr__()

    def pformat(self) -> str:
        return f'Struct({pformat(self.to_dict())})'

    def copy(
        self,
        update: Optional[dict] = None,

    ) -> msgspec.Struct:
        '''
        Validate-typecast all self defined fields, return a copy of us
        with all such fields.
        This is kinda like the default behaviour in `pydantic.BaseModel`.
        '''
        if update:
            for k, v in update.items():
                setattr(self, k, v)

        # roundtrip serialize to validate
        return msgspec.msgpack.Decoder(
            type=type(self)
        ).decode(
            msgspec.msgpack.Encoder().encode(self)
        )

    def typecast(
        self,
        # fields: Optional[list[str]] = None,
    ) -> None:
        for fname, ftype in self.__annotations__.items():
            setattr(self, fname, ftype(getattr(self, fname)))

# proto

class SkynetRPCRequest(Struct):
    uid: Union[str, int]  # user unique id
    method: str  # rpc method name
    params: dict  # variable params

class SkynetRPCResponse(Struct):
    result: dict

class ImageGenRequest(Struct):
    prompt: str
    step: int
    width: int
    height: int
    guidance: int
    seed: Optional[int]
    algo: str
    upscaler: Optional[str]

class DGPUBusRequest(Struct):
    rid: str  # req id
    nid: str  # node id
    task: str
    params: dict
