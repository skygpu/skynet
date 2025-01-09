docker build \
    -t guilledk/skynet:runtime-cuda-py311 \
    -f docker/Dockerfile.runtime+cuda-py311 .

docker build \
    -t guilledk/skynet:runtime-cuda \
    -f docker/Dockerfile.runtime+cuda-py311 .
