docker build \
    -t skynet:runtime-cuda \
    -f Dockerfile.runtime-cuda .

docker build \
    -t skynet:runtime \
    -f Dockerfile.runtime .
