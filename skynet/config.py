import os
import toml

import msgspec

from skynet.constants import DEFAULT_CONFIG_PATH, DEFAULT_IPFS_DOMAIN


class ConfigParsingError(BaseException):
    ...


class DgpuConfig(msgspec.Struct):
    account: str
    permission: str
    key: str
    node_url: str
    hyperion_url: str
    ipfs_url: str
    hf_token: str
    ipfs_domain: str = DEFAULT_IPFS_DOMAIN
    hf_home: str = 'hf_home'
    non_compete: set[str] = set()
    model_whitelist: set[str] = set() 
    model_blacklist: set[str] = set() 
    backend: str = 'sync-on-thread'
    api_bind: str = False
    tui: bool = False

class TelegramConfig(msgspec.Struct):
    account: str
    permission: str
    key: str
    node_url: str
    hyperion_url: str
    ipfs_url: str
    token: str

class DiscordConfig(msgspec.Struct):
    account: str
    permission: str
    key: str
    node_url: str
    hyperion_url: str
    ipfs_url: str
    token: str

class PinnerConfig(msgspec.Struct):
    hyperion_url: str
    ipfs_url: str

class UserConfig(msgspec.Struct):
    account: str
    permission: str
    key: str
    node_url: str

class Config(msgspec.Struct):
    dgpu: DgpuConfig | None = None
    telegram: TelegramConfig | None = None
    discord: DiscordConfig | None = None
    pinner: PinnerConfig | None = None
    user: UserConfig | None = None

def load_skynet_toml(file_path=DEFAULT_CONFIG_PATH) -> Config:
    with open(file_path, 'r') as file:
        return msgspec.toml.decode(file.read(), type=Config)


def set_hf_vars(hf_token: str, hf_home: str):
    os.environ['HF_TOKEN'] = hf_token
    os.environ['HF_HOME'] = hf_home
