docker build \
    -t guilledk/skynet:runtime-cuda-py312 \
    -f docker/Dockerfile.runtime+cuda-py312 . --progress=plain
