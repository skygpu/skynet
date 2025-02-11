from skynet.config import Config, DgpuConfig, set_config_override


def override_dgpu_config(**kwargs) -> DgpuConfig:
    config = Config(
        dgpu=DgpuConfig(**kwargs)
    )
    set_config_override(config)
    return config.dgpu
