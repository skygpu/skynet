# skynet
<div align="center">
    <img src="https://explorer.skygpu.net/v2/explore/assets/logo.png" width=512 height=512>
</div>
### decentralized compute platform

### native install

system dependencies:
- `cuda` 11.8
- `llvm` 10
- `python` 3.10+
- `docker` (for ipfs node)

```
# create and edit config from template
cp skynet.toml.example skynet.toml

# install poetry package manager
curl -sSL https://install.python-poetry.org | python3 -

# install
poetry install

# enable environment
poetry shell

# test you can run this command
skynet --help

# launch ipfs node
skynet run ipfs

# to launch worker
skynet run dgpu

```

### dockerized install

## frontend

system dependencies:
- `docker`

```
# create and edit config from template
cp skynet.toml.example skynet.toml

# pull runtime container
docker pull guilledk/skynet:runtime-frontend

# run telegram bot
docker run \
    -it \
    --rm \
    --network host \
    --name skynet-telegram \
    --mount type=bind,source="$(pwd)",target=/root/target \
    guilledk/skynet:runtime-frontend \
    skynet run telegram --db-pass PASSWORD --db-user USER --db-host HOST
```

## worker

system dependencies:
- `docker` with gpu enabled

```
# create and edit config from template
cp skynet.toml.example skynet.toml

# pull runtime container
docker pull guilledk/skynet:runtime-cuda

# or build it (takes a bit of time)
./build_docker.sh

# launch simple ipfs node
./launch_ipfs.sh

# run worker with all gpus
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    --name skynet-worker \
    --mount type=bind,source="$(pwd)",target=/root/target \
    guilledk/skynet:runtime-cuda \
    skynet run dgpu

# run worker with specific gpu
docker run \
    -it \
    --rm \
    --gpus '"device=1"' \
    --network host \
    --name skynet-worker-1 \
    --mount type=bind,source="$(pwd)",target=/root/target \
    guilledk/skynet:runtime-cuda \
    skynet run dgpu
```
