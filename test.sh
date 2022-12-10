docker run \
    -it \
    --rm \
    --mount type=bind,source="$(pwd)",target=/skynet \
    skynet:runtime-cuda \
    bash -c \
        "cd /skynet && pip install -e . && \
        pytest tests/test_dgpu.py --log-cli-level=info"
