#! /bin/bash -eu
mml_data=${1:-"mml-data"}
docker run \
-it \
-m=8000m \
--name mosaic_ml \
--mount type=bind,source="$(pwd)"/${mml_data},target=/home/developer/mosaic-ml/data \
alphasentaurii/mosaic-ml:latest