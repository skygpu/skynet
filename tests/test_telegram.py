#!/usr/bin/python

import trio

from functools import partial

from skynet.db import open_new_database
from skynet.brain import run_skynet
from skynet.config import load_skynet_ini
from skynet.frontend.telegram import run_skynet_telegram


if __name__ == '__main__':
    '''You will need a telegram bot token configured on skynet.ini for this
    '''
    with open_new_database() as db_params:
        db_container, db_pass, db_host = db_params
        config = load_skynet_ini()

        async def main():
            await run_skynet_telegram(
                'telegram-test',
                config['skynet.telegram-test']['token'],
                db_host=db_host,
                db_pass=db_pass
            )

        trio.run(main)
