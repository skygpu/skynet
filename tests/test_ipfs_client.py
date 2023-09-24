#!/usr/bin/python


from pathlib import Path


async def test_connection(ipfs_client):
    await ipfs_client.connect(
        '/ip4/169.197.140.154/tcp/4001/p2p/12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv')
    peers = await ipfs_client.peers()
    assert '12D3KooWKWogLFNEcNNMKnzU7Snrnuj84RZdMBg3sLiQSQc51oEv' in [p['Peer'] for p in peers]


async def test_add_and_pin_file(ipfs_client):
    test_file = Path('hello_world.txt')
    with open(test_file, 'w+') as file:
        file.write('Hello Skynet!')

    file_info = await ipfs_client.add(test_file)
    file_cid = file_info['Hash']

    pin_resp = await ipfs_client.pin(file_cid)

    assert file_cid in pin_resp

    test_file.unlink()
