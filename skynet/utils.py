#!/usr/bin/python

import time
import json
import hashlib


def hash_dict(d) -> str:
    d_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(d_str.encode('utf-8')).hexdigest()


def time_ms() -> int:
    return int(time.time() * 1000)
