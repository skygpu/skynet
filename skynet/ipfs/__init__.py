#!/usr/bin/python

import logging

import asks
import requests


class IPFSHTTP:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def pin(self, cid: str):
        return requests.post(
            f'{self.endpoint}/api/v0/pin/add',
            params={'arg': cid}
        )

    async def a_pin(self, cid: str):
        return await asks.post(
            f'{self.endpoint}/api/v0/pin/add',
            params={'arg': cid}
        )


async def get_ipfs_file(ipfs_link: str):
    logging.info(f'attempting to get image at {ipfs_link}')
    resp = None
    for i in range(10):
        try:
            resp = await asks.get(ipfs_link, timeout=3)

        except asks.errors.RequestTimeout:
            logging.warning('timeout...')

        except asks.errors.BadHttpResponse as e:
            logging.error(f'ifps gateway exception: \n{e}')

    if resp:
        logging.info(f'status_code: {resp.status_code}')
    else:
        logging.error(f'timeout')

    return resp
