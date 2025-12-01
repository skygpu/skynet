import os

import msgspec

from skynet.constants import DEFAULT_CONFIG_PATH, DEFAULT_IPFS_DOMAIN


class ConfigParsingError(BaseException):
    ...


class DgpuConfig(msgspec.Struct):
    account: str  # worker account name
    permission: str  # account permission name associated with key
    key: str  # private key
    node_url: str  # antelope http api endpoint
    ipfs_url: str  # IPFS node http rpc endpoint
    hf_token: str  # hugging face token
    ipfs_domain: str = DEFAULT_IPFS_DOMAIN  # IPFS Gateway domain
    hf_home: str = 'hf_home'  # hugging face data cache location
    non_compete: set[str] = set()  # set of worker names to not compete in requests
    model_whitelist: set[str] = set()  # only run these models
    model_blacklist: set[str] = set()  # don't run this models
    backend: str = 'sync-on-thread'  # select inference backend
    tui: bool = False  # enable TUI monitor
    poll_time: float = 0.5  # wait time for polling updates from contract
    log_level: str = 'info'
    log_file: str = 'dgpu.log'  # log file path (only used when tui = true)


class FrontendConfig(msgspec.Struct):
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
    telegram: FrontendConfig | None = None
    discord: FrontendConfig | None = None
    pinner: PinnerConfig | None = None
    user: UserConfig | None = None


__config_override = None
def set_config_override(config: Config):
    global __config_override
    __config_override = config


def load_skynet_toml(file_path=DEFAULT_CONFIG_PATH) -> Config:
    global __config_override

    if __config_override:
        return __config_override

    with open(file_path, 'r') as file:
        return msgspec.toml.decode(file.read(), type=Config)


def set_hf_vars(hf_token: str, hf_home: str):
    os.environ['HF_TOKEN'] = hf_token
    os.environ['HF_HOME'] = hf_home
