
docker build \
    -t guilledk/skynet:runtime \
    -f Dockerfile.runtime .

docker build \
    -t guilledk/skynet:runtime-frontend \
    -f Dockerfile.runtime+frontend .

docker build \
    -t guilledk/skynet:runtime-cuda-py311 \
    -f Dockerfile.runtime+cuda-py311 .

docker build \
    -t guilledk/skynet:runtime-cuda \
    -f Dockerfile.runtime+cuda-py311 .

docker build \
    -t guilledk/skynet:runtime-cuda-py310 \
    -f Dockerfile.runtime+cuda-py310 .
