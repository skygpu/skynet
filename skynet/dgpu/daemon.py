#!/usr/bin/python

import json
import logging
import traceback

from hashlib import sha256
from functools import partial

import trio

from skynet.dgpu.compute import SkynetMM
from skynet.dgpu.network import SkynetGPUConnector


class SkynetDGPUDaemon:

    def __init__(
        self,
        mm: SkynetMM,
        conn: SkynetGPUConnector,
        config: dict
    ):
        self.mm = mm
        self.conn = conn
        self.auto_withdraw = (
            config['auto_withdraw']
            if 'auto_withdraw' in config else False
        )

        self.non_compete = set()
        if 'non_compete' in config:
            self.non_compete = set(config['non_compete'])

        self.model_whitelist = set()
        if 'model_whitelist' in config:
            self.model_whitelist = set(config['model_whitelist'])

        self.model_blacklist = set()
        if 'model_blacklist' in config:
            self.model_blacklist = set(config['model_blacklist'])

        self.current_request = None

    async def should_cancel_work(self):
        competitors = set((
            status['worker']
            for status in
            (await self.conn.get_status_by_request_id(self.current_request))
        ))
        return self.non_compete & competitors

    async def serve_forever(self):
        try:
            while True:
                if self.auto_withdraw:
                    await self.conn.maybe_withdraw_all()

                queue = await self.conn.get_work_requests_last_hour()

                for req in queue:
                    rid = req['id']

                    # parse request
                    body = json.loads(req['body'])
                    model = body['params']['model']

                    # if whitelist enabled and model not in it continue
                    if (len(self.model_whitelist) > 0 and
                        not model in self.model_whitelist):
                        continue

                    # if blacklist contains model skip
                    if model in self.model_blacklist:
                        continue

                    my_results = [res['id'] for res in (await self.conn.find_my_results())]
                    if rid not in my_results:
                        statuses = await self.conn.get_status_by_request_id(rid)

                        if len(statuses) == 0:
                            self.current_request = rid

                            binary = await self.conn.get_input_data(req['binary_data'])

                            hash_str = (
                                str(req['nonce'])
                                +
                                req['body']
                                +
                                req['binary_data']
                            )
                            logging.info(f'hashing: {hash_str}')
                            request_hash = sha256(hash_str.encode('utf-8')).hexdigest()

                            # TODO: validate request

                            # perform work
                            logging.info(f'working on {body}')

                            resp = await self.conn.begin_work(rid)
                            if 'code' in resp:
                                logging.info(f'probably being worked on already... skip.')

                            else:
                                try:
                                    img_sha, img_raw = await trio.to_thread.run_sync(
                                        partial(
                                            self.mm.compute_one,
                                            self.should_cancel_work,
                                            body['method'], body['params'], binary=binary
                                        )
                                    )

                                    ipfs_hash = await self.conn.publish_on_ipfs(img_raw)

                                    await self.conn.submit_work(rid, request_hash, img_sha, ipfs_hash)
                                    break

                                except BaseException as e:
                                    traceback.print_exc()
                                    await self.conn.cancel_work(rid, str(e))
                                    break

                    else:
                        logging.info(f'request {rid} already beign worked on, skip...')

                await trio.sleep(1)

        except KeyboardInterrupt:
            ...
