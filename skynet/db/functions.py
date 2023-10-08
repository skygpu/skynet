#!/usr/bin/python

import time
import random
import string
import logging
import importlib

from datetime import datetime
from contextlib import contextmanager as cm
from contextlib import asynccontextmanager as acm

import docker
import asyncpg
import psycopg2

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ..constants import *


DB_INIT_SQL = '''
CREATE SCHEMA IF NOT EXISTS skynet;

CREATE TABLE IF NOT EXISTS skynet.user(
    id BIGSERIAL PRIMARY KEY NOT NULL,
    generated INT NOT NULL,
    joined TIMESTAMP NOT NULL,
    last_method TEXT,
    last_prompt TEXT,
    last_file   TEXT,
    last_binary TEXT,
    role VARCHAR(128) NOT NULL
);

CREATE TABLE IF NOT EXISTS skynet.user_config(
    id BIGSERIAL NOT NULL,
    model VARCHAR(512) NOT NULL,
    step INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    seed NUMERIC,
    guidance DECIMAL NOT NULL,
    strength DECIMAL NOT NULL,
    upscaler VARCHAR(128),
    autoconf BOOLEAN DEFAULT TRUE,
    CONSTRAINT fk_config
      FOREIGN KEY(id)
        REFERENCES skynet.user(id)
);

CREATE TABLE IF NOT EXISTS skynet.user_requests(
    id BIGSERIAL NOT NULL,
    user_id BIGSERIAL NOT NULL,
    sent TIMESTAMP NOT NULL,
    status TEXT NOT NULL,
    status_msg BIGSERIAL PRIMARY KEY NOT NULL,
    CONSTRAINT fk_user_req
      FOREIGN KEY(user_id)
        REFERENCES skynet.user(id)
);
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
        # could remove this if we ant the dockers to be persistent.
        # remove=True
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

        col_check = await conn.fetch(f'''
            select column_name 
            from information_schema.columns 
            where table_name = 'user_config' and column_name = 'autoconf';
        ''')

        if not col_check:
            await conn.execute('alter table skynet.user_config add column autoconf boolean default true;')

    async def _db_call(method: str, *args, **kwargs):
        method = getattr(db, method)

        async with pool.acquire() as conn:
            return await method(conn, *args, **kwargs)

    yield _db_call


async def get_user_config(conn, user: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user_config WHERE id = $1')
    conf = await stmt.fetch(user)
    if len(conf) == 1:
        return conf[0]

    else:
        return None


async def get_user(conn, uid: int):
    return await get_user_config(conn, uid)

async def get_last_method_of(conn, user: int):
    stmt = await conn.prepare(
        'SELECT last_method FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)

async def get_last_prompt_of(conn, user: int):
    stmt = await conn.prepare(
        'SELECT last_prompt FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)

async def get_last_file_of(conn, user: int):
    stmt = await conn.prepare(
        'SELECT last_file FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)

async def get_last_binary_of(conn, user: int):
    stmt = await conn.prepare(
        'SELECT last_binary FROM skynet.user WHERE id = $1')
    return await stmt.fetchval(user)


async def get_user_request(conn, mid: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user_requests WHERE id = $1')
    return await stmt.fetch(mid)

async def get_user_request_by_sid(conn, sid: int):
    stmt = await conn.prepare(
        'SELECT * FROM skynet.user_requests WHERE status_msg = $1')
    return (await stmt.fetch(sid))[0]

async def new_user_request(
    conn, user: int, mid: int,
    status_msg: int,
    status: str = 'started processing request...'
):
    date = datetime.utcnow()
    async with conn.transaction():
        stmt = await conn.prepare('''
            INSERT INTO skynet.user_requests(
                id, user_id, sent, status, status_msg
            )

            VALUES($1, $2, $3, $4, $5)
        ''')
        await stmt.fetch(mid, user, date, status, status_msg)

async def update_user_request(
    conn, mid: int, status: str
):
    stmt = await conn.prepare(f'''
        UPDATE skynet.user_requests
        SET status = $2
        WHERE id = $1
    ''')
    await stmt.fetch(mid, status)

async def update_user_request_by_sid(
    conn, sid: int, status: str
):
    stmt = await conn.prepare(f'''
        UPDATE skynet.user_requests
        SET status = $2
        WHERE status_msg = $1
    ''')
    await stmt.fetch(sid, status)


async def new_user(conn, uid: int):
    if await get_user(conn, uid):
        raise ValueError('User already present on db')

    logging.info(f'new user! {uid}')

    date = datetime.utcnow()
    async with conn.transaction():
        stmt = await conn.prepare('''
            INSERT INTO skynet.user(
                id, generated, joined,
                last_method, last_prompt, last_file, last_binary,
                role
            )

            VALUES($1, $2, $3, $4, $5, $6, $7, $8)
        ''')
        await stmt.fetch(
            uid, 0, date, 'txt2img', None, None, None, DEFAULT_ROLE
        )

        stmt = await conn.prepare('''
            INSERT INTO skynet.user_config(
                id, model, step, width, height, guidance, strength, upscaler)

            VALUES($1, $2, $3, $4, $5, $6, $7, $8)
        ''')
        resp = await stmt.fetch(
            uid,
            DEFAULT_MODEL,
            DEFAULT_STEP,
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            DEFAULT_GUIDANCE,
            DEFAULT_STRENGTH,
            DEFAULT_UPSCALER
        )


async def get_or_create_user(conn, uid: str):
    user = await get_user(conn, uid)

    if not user:
        await new_user(conn, uid)
        user = await get_user(conn, uid)

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

async def increment_generated(conn, user: int):
    stmt = await conn.prepare('''
        UPDATE skynet.user
        SET generated = generated + 1
        WHERE id = $1
    ''')
    await stmt.fetch(user)

async def update_user_stats(
    conn,
    user: int,
    method: str,
    last_prompt: str | None = None,
    last_file: str | None = None,
    last_binary: str | None = None
):
    await update_user(conn, user, 'last_method', method)
    if last_prompt:
        await update_user(conn, user, 'last_prompt', last_prompt)
    if last_file:
        await update_user(conn, user, 'last_file', last_file)
    if last_binary:
        await update_user(conn, user, 'last_binary', last_binary)

    logging.info((method, last_prompt, last_binary))
