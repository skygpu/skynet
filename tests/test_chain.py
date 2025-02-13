import trio
import pytest
from leap.errors import TransactionPushError
from msgspec import json

from skynet.contract import RequestNotFound, ConfigNotFound
from skynet.types import BodyV0, BodyV0Params
from skynet._testing import open_test_worker

async def test_system(skynet_cleos):
    gpu, cleos = skynet_cleos

    # assert config can only be init once (done by fixture)
    with pytest.raises(TransactionPushError):
        await gpu.init_config('eosio.token', '4,TLOS')

    # test clean function

    # fill tables with data

    # accounts
    users = []
    quantity = '1.0000 TLOS'
    usr_num = 3
    for _ in range(usr_num):
       test_user = cleos.new_account()
       cleos.transfer_token('eosio', test_user, quantity, 'clean testing')
       await gpu.deposit(test_user, quantity)
       # will throw if user not found
       await gpu.get_user(test_user)
       users.append(test_user)

    # queue
    for user in users:
        await gpu.enqueue(
            user,
            BodyV0(
                method='txt2img',
                params=BodyV0Params(
                    prompt='ayy lmao',
                    model='skygpu/mocker',
                    step=1,
                    seed=0,
                    guidance=10.0
                )
            ),
            min_verification=2  # make reqs stay after 1 result
        )

    # check requests are in queue
    queue = await gpu.get_queue()
    assert len(queue) == usr_num

    # workers
    workers = []
    quantity = '1.0000 TLOS'
    wrk_num = 3
    for _ in range(wrk_num):
       worker = cleos.new_account()
       await gpu.register_worker(worker, 'http://localhost')
       # will throw if worker not found
       await gpu.get_worker(worker)
       workers.append(worker)

    # status
    for i in range(wrk_num):
        req = queue[i]
        worker = workers[i]
        await gpu.accept_work(worker, req.id)
        # will throw is status not found
        await gpu.get_worker_status_for_request(req.id, worker)

    # results
    # make one of the workers finish to populate result
    await gpu.submit_work(
        workers[0],
        queue[0].id,
        'ff' * 32,
        'null hash'
    )

    results = await gpu.get_results(queue[0].id)
    assert len(results) == 1

    # all tables populated

    # run clean nuke == false
    await gpu.clean_tables()

    # assert tables empty
    assert len(await gpu.get_queue()) == 0
    for req in queue:
        assert len(await gpu.get_statuses_for_request(req.id)) == 0
        assert len(await gpu.get_results(req.id)) == 0

    # check config, accounts and workers still there
    await gpu.get_config()  # raises if not found
    assert len(await gpu.get_users()) == usr_num
    assert len(await gpu.get_workers()) == wrk_num

    # test nuke
    await gpu.clean_tables(nuke=True)
    with pytest.raises(ConfigNotFound):
        await gpu.get_config()

    assert len(await gpu.get_users()) == 0
    assert len(await gpu.get_workers()) == 0

    # re init config in case other tests run
    await gpu.init_config('eosio.token', '4,TLOS')


async def test_balance(skynet_cleos):
    gpu, cleos = skynet_cleos

    # create fresh account
    account = cleos.new_account()

    # try call withdraw with no user account reg'd
    with pytest.raises(TransactionPushError):
        await gpu.withdraw(account, '1.0000 TLOS')

    # give tokens and deposit to gpu
    quantity = '1000.0000 TLOS'
    cleos.transfer_token('eosio', account, quantity)
    await gpu.deposit(account, quantity)

    # check if balance increased
    account_row = await gpu.get_user(account)
    assert account_row.balance == quantity

    # try call withdraw with more than deposited
    with pytest.raises(TransactionPushError):
        await gpu.withdraw(account, '1000.0001 TLOS')

    # withdraw full correct amount
    await gpu.withdraw(account, quantity)

    # check if balance decreased
    account_row = await gpu.get_user(account)
    assert account_row.balance == '0.0000 TLOS'

async def test_worker_reg(skynet_cleos):
    gpu, cleos = skynet_cleos

    # create fresh account
    worker = cleos.new_account()
    url = 'https://nvidia.com'

    await gpu.register_worker(worker, url)

    # find and check vals
    worker_row = await gpu.get_worker(worker)
    assert worker_row.account == worker
    assert worker_row.url == url
    assert worker_row.joined != '1970-01-01T00:00:00'
    assert worker_row.left == '1970-01-01T00:00:00'

    # attempt to register twice
    with pytest.raises(TransactionPushError):
        await gpu.register_worker(worker, url)

    # unregister
    reason = 'testing'
    await gpu.unregister_worker(worker, reason)

    worker_row = await gpu.get_worker(worker)
    assert worker_row.account == worker
    assert worker_row.url == url
    assert worker_row.left != '1970-01-01T00:00:00'

    # attempt to unreg twice
    with pytest.raises(TransactionPushError):
        await gpu.unregister_worker(worker, reason)


async def test_queue(skynet_cleos):
    gpu, cleos = skynet_cleos

    body = BodyV0(
        method='txt2img',
        params=BodyV0Params(
            prompt='cyberpunk hacker travis bickle dystopic alley graffiti',
            model='skygpu/mocker',
            step=4,
            seed=0,
            guidance=10.0
        )
    )

    # create account
    account = cleos.new_account()
    quantity = '1000.0000 TLOS'
    cleos.transfer_token('eosio', account, quantity)

    # attempt to create request without prev deposit
    with pytest.raises(TransactionPushError):
        await gpu.enqueue(account, body)

    # deposit tokens into gpu
    await gpu.deposit(account, quantity)

    # finally enqueue
    req = await gpu.enqueue(account, body)

    # search by id
    req_found = await gpu.get_request(req.id)
    assert req == req_found

    # search by timestamp
    reqs = await gpu.get_requests_since(60 * 60)
    assert len(reqs) == 1
    assert reqs[0] == req

    # attempt to dequeue wrong req
    with pytest.raises(TransactionPushError):
        await gpu.dequeue(account, 999999)

    # dequeue correctly
    await gpu.dequeue(account, req.id)

    # check deletion
    with pytest.raises(RequestNotFound):
        await gpu.get_request(req.id)


async def test_full_flow(inject_mockers, skynet_cleos, ipfs_node):
    gpu, cleos = skynet_cleos

    # create account and deposit tokens into gpu
    account = cleos.new_account()
    quantity = '1000.0000 TLOS'
    cleos.transfer_token('eosio', account, quantity)
    cleos.transfer_token(account, 'gpu.scd', quantity)

    og_body = BodyV0(
        method='txt2img',
        params=BodyV0Params(
            prompt='cyberpunk hacker travis bickle dystopic alley graffiti',
            model='skygpu/mocker',
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
