import trio
from msgspec import json

from skynet.types import BodyV0, BodyV0Params

from skynet._testing import open_test_worker


async def test_full_flow(inject_mockers, skynet_cleos, ipfs_node):
    cleos = skynet_cleos

    # create account and deposit tokens into gpu
    account = cleos.new_account()
    quantity = '1000.0000 TLOS'
    cleos.transfer_token('eosio', account, quantity)
    cleos.transfer_token(account, 'gpu.scd', quantity)

    og_body = BodyV0(
        method='txt2img',
        params=BodyV0Params(
            prompt='cyberpunk hacker travis bickle dystopic alley graffiti',
            model='skygpu/txt2img-mocker',
            step=4,
            seed=0,
            guidance=10.0
        )
    )

    body = json.encode(og_body).decode('utf-8')
    binary_data = ''
    reward = '1.0000 TLOS'
    min_verification = 1

    # send enqueue req
    cleos.push_action(
        'gpu.scd',
        'enqueue',
        [
            account,
            body,
            binary_data,
            reward,
            min_verification
        ],
        account,
        key=cleos.private_keys[account]
    )

    # open worker and fill request
    async with open_test_worker(cleos, ipfs_node) as (_conn, state_mngr):
        while state_mngr.queue_len > 0:
            await trio.sleep(1)
