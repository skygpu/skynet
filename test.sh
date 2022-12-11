docker run \
    -it \
    --rm \
    --gpus=all \
    --mount type=bind,source="$(pwd)",target=/skynet \
    skynet:runtime-cuda \
    bash -c \
        "cd /skynet && pip install -e . && \
        pytest $1 --log-cli-level=info"
