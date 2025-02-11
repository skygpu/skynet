from contextlib import asynccontextmanager as acm

from skynet.config import Config, DgpuConfig, set_config_override
from skynet.dgpu import open_worker

def override_dgpu_config(**kwargs) -> DgpuConfig:
    config = Config(
        dgpu=DgpuConfig(**kwargs)
    )
    set_config_override(config)
    return config.dgpu


@acm
async def open_test_worker(
    cleos, ipfs_node,
    account: str = 'testworker',
    permission: str = 'active',
    hf_token: str = '',
    **kwargs
):
    config = override_dgpu_config(
        account=account,
        permission=permission,
        key=cleos.private_keys[account],
        node_url=cleos.endpoint,
        ipfs_url=ipfs_node[1].endpoint,
        hf_token=hf_token,
        **kwargs
    )
    async with open_worker(config) as worker:
        yield worker
