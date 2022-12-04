docker run \
    -it \
    --rm \
    --gpus=all \
    --env HF_TOKEN='' \
    --env DB_USER='' \
    --env DB_PASS='' \
    --mount type=bind,source="$(pwd)"/outputs,target=/outputs \
    --mount type=bind,source="$(pwd)"/hf_home,target=/hf_home \
    --mount type=bind,source="$(pwd)"/scripts,target=/scripts \
    skynet-art-bot:0.1a3 $1
