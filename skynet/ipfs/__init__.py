import io
import logging
from pathlib import Path

import httpx
from PIL import Image

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

    async def publish(self, raw, type: str = 'png'):
        stage = Path('/tmp/ipfs-staging')
        stage.mkdir(exist_ok=True)
        logging.info('publish_on_ipfs')

        target_file = ''
        match type:
            case 'png':
                raw: Image
                target_file = stage / 'image.png'
                raw.save(target_file)

            case _:
                raise ValueError(f'Unsupported output type: {type}')

        file_info = await self.add(Path(target_file))
        file_cid = file_info['Hash']
        logging.info(f'added file to ipfs, CID: {file_cid}')

        await self.pin(file_cid)
        logging.info(f'pinned {file_cid}')

        return file_cid

async def get_ipfs_img(ipfs_link: str, timeout: int = 3) -> Image:
    logging.info(f'attempting to get image at {ipfs_link}')
    resp = None
    for _ in range(timeout):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(ipfs_link, timeout=timeout)

        except httpx.RequestError as e:
            logging.warning(f'Request error: {e}')

        if resp is not None:
            break

    if resp:
        logging.info(f'status_code: {resp.status_code}')
    else:
        logging.error('timeout')
        return None

    if resp.status_code != 200:
        logging.warning(f'couldn\'t get ipfs binary data at {ipfs_link}!')
        return resp

    img = Image.open(io.BytesIO(resp.read()))
    logging.info('decoded img successfully')

    return img
