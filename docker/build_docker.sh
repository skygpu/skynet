
docker build \
    -t guilledk/skynet:runtime \
    -f docker/Dockerfile.runtime .

docker build \
    -t guilledk/skynet:runtime-frontend \
    -f docker/Dockerfile.runtime+frontend .

docker build \
    -t guilledk/skynet:runtime-cuda-py311 \
    -f docker/Dockerfile.runtime+cuda-py311 .

docker build \
    -t guilledk/skynet:runtime-cuda \
    -f docker/Dockerfile.runtime+cuda-py311 .

docker build \
    -t guilledk/skynet:runtime-cuda-py310 \
    -f docker/Dockerfile.runtime+cuda-py310 .
