#!/usr/bin/python

import logging

from datetime import datetime
from contextlib import asynccontextmanager as acm

import trio
import triopg

from .constants import *


def try_decode_uid(uid: str):
    try:
        proto, uid = uid.split('+')
        uid = int(uid)
        return proto, uid

    except ValueError:
        logging.warning(f'got non numeric uid?: {uid}')
        return None, None


@acm
async def open_database_connection(
    db_user: str,
    db_pass: str,
    db_host: str = DB_HOST,
):
    async with triopg.create_pool(
        dsn=f'postgres://{db_user}:{db_pass}@{db_host}/skynet_art_bot'
    ) as conn:
        yield conn


async def get_user(conn, uid: str):
    if isinstance(uid, str):
        proto, uid = try_decode_uid(uid)

        match proto:
            case 'tg':
                stmt = await conn.prepare(
                    'SELECT * FROM skynet.user WHERE tg_id = $1')
                user = await stmt.fetchval(uid)

            case _:
                user = None

        return user

    else:  # asumme is our uid
        stmt = await conn.prepare(
            'SELECT * FROM skynet.user WHERE id = $1')
        return await stmt.fetchval(uid)


async def get_user_config(conn, user: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user_config WHERE id = $1')
    return (await stmt.fetch(user))[0]


async def get_last_prompt_of(conn, user: int):
    stms = await conn.prepare(
        'SELECT last_prompt FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)


async def new_user(conn, uid: str):
    if await get_user(conn, uid):
        raise ValueError('User already present on db')

    logging.info(f'new user! {uid}')

    tg_id = None
    date = datetime.utcnow()

    proto, pid = try_decode_uid(uid)

    match proto:
        case 'tg':
            tg_id = pid

    async with conn.transaction():
        stmt = await conn.prepare('''
            INSERT INTO skynet.user(
                tg_id, generated, joined, last_prompt, role)

            VALUES($1, $2, $3, $4, $5)
        ''')
        await stmt.fetch(
            tg_id, 0, date, None, DEFAULT_ROLE
        )

        new_uid = await get_user(conn, uid)

        stmt = await conn.prepare('''
            INSERT INTO skynet.user_config(
                id, algo, step, width, height, seed, guidance, upscaler)

            VALUES($1, $2, $3, $4, $5, $6, $7, $8)
        ''')
        user = await stmt.fetch(
            new_uid,
            DEFAULT_ALGO,
            DEFAULT_STEP,
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            DEFAULT_SEED,
            DEFAULT_GUIDANCE,
            DEFAULT_UPSCALER
        )

    return new_uid


async def get_or_create_user(conn, uid: str):
    user = await get_user(conn, uid)

    if not user:
        user = await new_user(conn, uid)

    return user

async def update_user(conn, user: int, attr: str, val):
    ...

async def update_user_config(conn, user: int, attr: str, val):
    stmt = await conn.prepare(f'''
        UPDATE skynet.user_config
        SET {attr} = $2
        WHERE id = $1
    ''')
    await stmt.fetch(user, val)


async def get_user_stats(conn, user: int):
    stmt = await conn.prepare('''
        SELECT generated,joined,role FROM skynet.user
        WHERE id = $1
    ''')
    records = await stmt.fetch(user)
    assert len(records) == 1
    record = records[0]
    return record