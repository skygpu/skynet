#!/bin/bash

name='skynet-ipfs'
peers=("$@")

data_dir="$(pwd)/ipfs-docker-data"
data_target='/data/ipfs'

# Create data directory if it doesn't exist
mkdir -p "$data_dir"

# Run the container
docker run -d \
  --name "$name" \
  -p 8080:8080/tcp \
  -p 4001:4001/tcp \
  -p 127.0.0.1:5001:5001/tcp \
  --mount type=bind,source="$data_dir",target="$data_target" \
  --rm \
  ipfs/go-ipfs:latest

# Change ownership
docker exec "$name" chown 1000:1000 -R "$data_target"

# Wait for Daemon to be ready
while read -r log; do
  echo "$log"
  if [[ "$log" == *"Daemon is ready"* ]]; then
    break
  fi
done < <(docker logs -f "$name")

# Connect to peers
for peer in "${peers[@]}"; do
  docker exec "$name" ipfs swarm connect "$peer" || echo "Error connecting to peer: $peer"
done
