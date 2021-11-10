#! /bin/bash -eu
export MML_DOCKER_IMAGE=alphasentaurii/mosaic-ml:latest
export CAL_BASE_IMAGE="stsci/hst-pipeline:CALDP_drizzlecats_CAL_rc5"
docker build -f Dockerfile -t ${MML_DOCKER_IMAGE} --build-arg CAL_BASE_IMAGE="${CAL_BASE_IMAGE}" .
