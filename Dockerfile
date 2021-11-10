# Copyright (c) Association of Universities for Research in Astronomy
# Distributed under the terms of the Modified BSD License.
# DATB's HST CAL code build for fundamental calibration s/w
ARG CAL_BASE_IMAGE="stsci/hst-pipeline:CALDP_drizzlecats_CAL_rc5"
FROM ${CAL_BASE_IMAGE}

LABEL maintainer="dmd_octarine@stsci.edu" \
      vendor="Space Telescope Science Institute"

# Environment variables
ENV MKL_THREADING_LAYER="GNU"


USER root

RUN yum remove -y kernel-devel   &&\
 yum update  -y && \
 yum install -y \
   emacs-nox \
   make \
   gcc \
   gcc-c++ \
   gcc-gfortran \
   python3 \
   python3-devel \
   htop \
   wget \
   git \
   libpng-devel \
   libjpeg-devel \
   libcurl-devel \
   tar \
   patch \
   curl \
   rsync \
   time

WORKDIR /home/developer
RUN mkdir /home/developer/mosaic-ml
ADD mosaic-ml/ /home/developer/mosaic-ml/
RUN chown -R developer:developer /home/developer

COPY requirements.txt /home/developer/.
RUN python -m pip install --upgrade pip && python -m pip install -r /home/developer/requirements.txt


# ------------------------------------------------
USER developer
ENV SVM_QUALITY_TESTING=on
#RUN conda init bash && source ~/.bashrc
#RUN cd mosaic-ml  &&  pip install .

CMD ["/bin/bash"]

