#!/usr/bin/python

import json
import logging

from hashlib import sha256
from collections import OrderedDict

from google.protobuf.json_format import MessageToDict
from OpenSSL.crypto import PKey, X509, verify, sign

from .skynet_pb2 import *


def serialize_msg_deterministic(msg):
    descriptors = sorted(
        type(msg).DESCRIPTOR.fields_by_name.items(),
        key=lambda x: x[0]
    )
    shasum = sha256()

    def hash_dict(d):
        data = [
            (key, val)
            for (key, val) in d.items()
        ]
        for key, val in sorted(data, key=lambda x: x[0]):
            if not isinstance(val, dict):
                shasum.update(key.encode())
                shasum.update(json.dumps(val).encode())
            else:
                hash_dict(val)

    for (field_name, field_descriptor) in descriptors:
        if not field_descriptor.message_type:
            shasum.update(field_name.encode())

            value = getattr(msg, field_name)

            if isinstance(value, bytes):
                value = value.hex()

            shasum.update(json.dumps(value).encode())
            continue

        if field_descriptor.message_type.name == 'Struct':
            hash_dict(MessageToDict(getattr(msg, field_name)))

    deterministic_msg = shasum.hexdigest()

    return deterministic_msg


def sign_protobuf_msg(msg, key: PKey):
    return sign(
        key, serialize_msg_deterministic(msg), 'sha256').hex()


def verify_protobuf_msg(msg, cert: X509):
    return verify(
        cert,
        bytes.fromhex(msg.auth.sig),
        serialize_msg_deterministic(msg),
        'sha256'
    )
