docker run
    -d \
    --rm \
    -p 27017:27017 \
	--name mongodb-skynet \
    --mount type=bind,source="$(pwd)"/mongodb,target=/data/db \
	-e MONGO_INITDB_ROOT_USERNAME="" \
	-e MONGO_INITDB_ROOT_PASSWORD="" \
    mongo	
