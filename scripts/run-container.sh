#! /bin/bash -eu
docker run \
-it \
-m=8000m \
--name mosaicml \
--mount type=bind,source="$(pwd)"/mml-data,target=/home/developer/mosaic-ml/data \
alphasentaurii/mosaic-ml:latest