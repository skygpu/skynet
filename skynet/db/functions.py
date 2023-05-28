#!/usr/bin/python

import time
import random
import string
import logging
import importlib

from typing import Optional
from datetime import datetime
from contextlib import contextmanager as cm
from contextlib import asynccontextmanager as acm

import docker
import asyncpg
import psycopg2

from asyncpg.exceptions import UndefinedColumnError
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ..constants import *


DB_INIT_SQL = '''
CREATE SCHEMA IF NOT EXISTS skynet;

CREATE TABLE IF NOT EXISTS skynet.user(
   id SERIAL PRIMARY KEY NOT NULL,
   tg_id BIGINT,
   wp_id VARCHAR(128),
   mx_id VARCHAR(128),
   ig_id VARCHAR(128),
   generated INT NOT NULL,
   joined DATE NOT NULL,
   last_prompt TEXT,
   role VARCHAR(128) NOT NULL
);
ALTER TABLE skynet.user
    ADD CONSTRAINT tg_unique
    UNIQUE (tg_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT wp_unique
    UNIQUE (wp_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT mx_unique
    UNIQUE (mx_id);
ALTER TABLE skynet.user
    ADD CONSTRAINT ig_unique
    UNIQUE (ig_id);

CREATE TABLE IF NOT EXISTS skynet.user_config(
    id SERIAL NOT NULL,
    algo VARCHAR(128) NOT NULL,
    step INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    seed BIGINT NOT NULL,
    guidance REAL NOT NULL,
    strength REAL NOT NULL,
    upscaler VARCHAR(128)
);
ALTER TABLE skynet.user_config
    ADD FOREIGN KEY(id)
    REFERENCES skynet.user(id);
'''


def try_decode_uid(uid: str):
    try:
        return None, int(uid)
    except ValueError:
        ...

    try:
        proto, uid = uid.split('+')
        uid = int(uid)
        return proto, uid

    except ValueError:
        logging.warning(f'got non chat proto uid?: {uid}')
        return None, None


@cm
def open_new_database(cleanup=True):
    rpassword = ''.join(
        random.choice(string.ascii_lowercase)
        for i in range(12))
    password = ''.join(
        random.choice(string.ascii_lowercase)
        for i in range(12))

    dclient = docker.from_env()

    container = dclient.containers.run(
        'postgres',
        name='skynet-test-postgres',
        ports={'5432/tcp': None},
        environment={
            'POSTGRES_PASSWORD': rpassword
        },
        detach=True,
        remove=True
    )
    try:

        for log in container.logs(stream=True):
            log = log.decode().rstrip()
            logging.info(log)
            if ('database system is ready to accept connections' in log or
                'database system is shut down' in log):
                break

        # ip = container.attrs['NetworkSettings']['IPAddress']
        container.reload()
        port = container.ports['5432/tcp'][0]['HostPort']
        host = f'localhost:{port}'

        # why print the system is ready to accept connections when its not
        # postgres? wtf
        time.sleep(1)
        logging.info('creating skynet db...')

        conn = psycopg2.connect(
            user='postgres',
            password=rpassword,
            host='localhost',
            port=port
        )
        logging.info('connected...')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(
                f'CREATE USER skynet WITH PASSWORD \'{password}\'')
            cursor.execute(
                f'CREATE DATABASE skynet')
            cursor.execute(
                f'GRANT ALL PRIVILEGES ON DATABASE skynet TO skynet')

        conn.close()

        logging.info('done.')
        yield container, password, host

    finally:
        if container and cleanup:
            container.stop()

@acm
async def open_database_connection(
    db_user: str = 'skynet',
    db_pass: str = 'password',
    db_host: str = 'localhost:5432',
    db_name: str = 'skynet'
):
    db = importlib.import_module('skynet.db.functions')
    pool = await asyncpg.create_pool(
        dsn=f'postgres://{db_user}:{db_pass}@{db_host}/{db_name}')

    async with pool.acquire() as conn:
        res = await conn.execute(f'''
            select distinct table_schema
            from information_schema.tables
            where table_schema = \'{db_name}\'
        ''')
        if '1' in res:
            logging.info('schema already in db, skipping init')
        else:
            await conn.execute(DB_INIT_SQL)

    async def _db_call(method: str, *args, **kwargs):
        method = getattr(db, method)

        async with pool.acquire() as conn:
            return await method(conn, *args, **kwargs)

    yield _db_call


async def get_user(conn, uid: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(uid)


async def get_user_config(conn, user: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user_config WHERE id = $1')
    return (await stmt.fetch(user))[0]


async def get_last_prompt_of(conn, user: int):
    stmt = await conn.prepare(
        'SELECT last_prompt FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)


async def new_user(conn, uid: int):
    if await get_user(conn, uid):
        raise ValueError('User already present on db')

    logging.info(f'new user! {uid}')

    date = datetime.utcnow()
    async with conn.transaction():
        stmt = await conn.prepare('''
            INSERT INTO skynet.user(
                id, generated, joined, last_prompt, role)

            VALUES($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
        ''')
        await stmt.fetch(
            uid, 0, date, None, DEFAULT_ROLE
        )

        stmt = await conn.prepare('''
            INSERT INTO skynet.user_config(
                id, algo, step, width, height, seed, guidance, strength, upscaler)

            VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
        ''')
        user = await stmt.fetch(
            new_uid,
            DEFAULT_ALGO,
            DEFAULT_STEP,
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            DEFAULT_SEED,
            DEFAULT_GUIDANCE,
            DEFAULT_STRENGTH,
            DEFAULT_UPSCALER
        )


async def get_or_create_user(conn, uid: str):
    user = await get_user(conn, uid)

    if not user:
        user = await new_user(conn, uid)

    return user

async def update_user(conn, user: int, attr: str, val):
    stmt = await conn.prepare(f'''
        UPDATE skynet.user
        SET {attr} = $2
        WHERE id = $1
    ''')
    await stmt.fetch(user, val)

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

async def update_user_stats(
    conn,
    user: int,
    last_prompt: Optional[str] = None
):
    stmt = await conn.prepare('''
        UPDATE skynet.user
        SET generated = generated + 1
        WHERE id = $1
    ''')
    await stmt.fetch(user)

    if last_prompt:
        await update_user(conn, user, 'last_prompt', last_prompt)
