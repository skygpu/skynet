#!/usr/bin/python

import logging
from pathlib import Path

import httpx


class IPFSClientException(Exception):
    ...


class AsyncIPFSHTTP:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def _post(self, sub_url: str, *args, **kwargs):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.endpoint + sub_url,
                *args, **kwargs
            )

        if resp.status_code != 200:
            raise IPFSClientException(resp.text)

        return resp.json()
#!/usr/bin/python
    async def add(self, file_path: Path, **kwargs):
        files = {
            'file': (file_path.name, file_path.open('rb'))
        }
        return await self._post(
            '/api/v0/add',
            files=files,
            params=kwargs
        )

    async def pin(self, cid: str):
        return (await self._post(
            '/api/v0/pin/add',
            params={'arg': cid}
        ))['Pins']

    async def connect(self, multi_addr: str):
        return await self._post(
            '/api/v0/swarm/connect',
            params={'arg': multi_addr}
        )

    async def peers(self, **kwargs):
        return (await self._post(
            '/api/v0/swarm/peers',
            params=kwargs
        ))['Peers']


async def get_ipfs_file(ipfs_link: str, timeout: int = 60 * 5):
    logging.info(f'attempting to get image at {ipfs_link}')
    resp = None
    for _ in range(timeout):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(ipfs_link, timeout=3)

        except httpx.RequestError as e:
            logging.warning(f'Request error: {e}')

        if resp is not None:
            break

    if resp:
        logging.info(f'status_code: {resp.status_code}')
    else:
        logging.error(f'timeout')

    return resp
