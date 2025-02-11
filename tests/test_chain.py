from msgspec import json

from skynet.types import BodyV0, BodyV0Params
from skynet.dgpu.network import NetConnector

from skynet._testing import override_dgpu_config


async def test_enqueue(skynet_cleos):
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

    config = override_dgpu_config(
        account='testworker1',
        permission='active',
        key='',
        node_url=cleos.endpoint,
        ipfs_url='http://127.0.0.1:5001',
        hf_token=''
    )
    net = NetConnector(config)
    queue = await net.get_work_requests_last_hour()

    assert len(queue) == 1

    req = queue[0]
    body = json.decode(req.body, type=BodyV0)

    assert og_body == body
