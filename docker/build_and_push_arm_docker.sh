#!/bin/bash

TAG=$1
PLATFORM=$2

docker buildx build --platform $PLATFORM --push -t registry.gitlab.com/skails/geniusai-server:$TAG -f ./docker/Dockerfile .