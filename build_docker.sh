docker build \
    -t skynet:runtime-cuda \
    -f docker/Dockerfile.runtime+cuda .

docker build \
    -t skynet:runtime \
    -f docker/Dockerfile.runtime .
