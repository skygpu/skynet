#!/usr/bin/python

import importlib

from contextlib import asynccontextmanager as acm

import trio
import tractor
import asyncpg
import asyncio
import trio_asyncio


_spawn_kwargs = {
    'infect_asyncio': True,
}


async def aio_db_proxy(
    to_trio: trio.MemorySendChannel,
    from_trio: asyncio.Queue,
    db_user: str = 'skynet',
    db_pass: str = 'password',
    db_host: str = 'localhost:5432',
    db_name: str = 'skynet'
) -> None:
    db = importlib.import_module('skynet.db.functions')

    pool = await asyncpg.create_pool(
        dsn=f'postgres://{db_user}:{db_pass}@{db_host}/{db_name}')

    async with pool_conn.acquire() as conn:
        res = await conn.execute(f'''
            select distinct table_schema
            from information_schema.tables
            where table_schema = \'{db_name}\'
        ''')
        if '1' in res:
            logging.info('schema already in db, skipping init')
        else:
            await conn.execute(DB_INIT_SQL)

    # a first message must be sent **from** this ``asyncio``
    # task or the ``trio`` side will never unblock from
    # ``tractor.to_asyncio.open_channel_from():``
    to_trio.send_nowait('start')

    # XXX: this uses an ``from_trio: asyncio.Queue`` currently but we
    # should probably offer something better.
    while True:
        msg = await from_trio.get()

        method = getattr(db, msg.get('method'))
        args = getattr(db, msg.get('args', []))
        kwargs = getattr(db, msg.get('kwargs', {}))

        async with pool_conn.acquire() as conn:
            result = await method(conn, *args, **kwargs)
            to_trio.send_nowait(result)


@tractor.context
async def trio_to_aio_db_proxy(
    ctx: tractor.Context,
    db_user: str = 'skynet',
    db_pass: str = 'password',
    db_host: str = 'localhost:5432',
    db_name: str = 'skynet'
):
    # this will block until the ``asyncio`` task sends a "first"
    # message.
    async with tractor.to_asyncio.open_channel_from(
        aio_db_proxy,
        db_user=db_user,
        db_pass=db_pass,
        db_host=db_host,
        db_name=db_name
    ) as (first, chan):

        assert first == 'start'
        await ctx.started(first)

        async with ctx.open_stream() as stream:

            async for msg in stream:
                await chan.send(msg)

                out = await chan.receive()
                # echo back to parent actor-task
                await stream.send(out)


@acm
async def open_database_connection(
    db_user: str = 'skynet',
    db_pass: str = 'password',
    db_host: str = 'localhost:5432',
    db_name: str = 'skynet'
):
    async with tractor.open_nursery() as n:
        p = await n.start_actor(
            'aio_db_proxy',
            enable_modules=[__name__],
            infect_asyncio=True,
        )
        async with p.open_context(
            trio_to_aio_db_proxy,
            db_user=db_user,
            db_pass=db_pass,
            db_host=db_host,
            db_name=db_name
        ) as (ctx, first):
            async with ctx.open_stream() as stream:

                async def _db_pc(method: str, *args, **kwargs):
                    await stream.send({
                        'method': method,
                        'args': args,
                        'kwargs': kwargs
                    })
                    return await stream.receive()

                yield _db_pc
