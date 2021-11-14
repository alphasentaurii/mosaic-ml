#! /bin/bash -eu
mml_data=${1:-"mml-data"}
docker run \
-it \
-m=8000m \
--name mosaicml \
--mount type=bind,source="$(pwd)"/${mml_data},target=/home/developer/mosaic-ml/data \
alphasentaurii/mosaic-ml:latest