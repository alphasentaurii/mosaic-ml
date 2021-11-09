# Copyright (c) Association of Universities for Research in Astronomy
# Distributed under the terms of the Modified BSD License.
# DATB's HST CAL code build for fundamental calibration s/w
ARG CAL_BASE_IMAGE="stsci/hst-pipeline:CALDP_drizzlecats_CAL_rc5"
FROM ${CAL_BASE_IMAGE}

LABEL maintainer="dmd_octarine@stsci.edu" \
      vendor="Space Telescope Science Institute"

# Environment variables
ENV MKL_THREADING_LAYER="GNU"
ENV REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ENV CURL_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
# ------------------------------------------------------------------------
# SSL/TLS cert setup for STScI AWS firewalling

USER root
# RUN mkdir -p /etc/ssl/certs && \
#     mkdir -p /etc/pki/ca-trust/extracted/pem
# COPY tls-ca-bundle.pem /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
# RUN mv /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt.org && \
#     ln -s /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem  /etc/ssl/certs/ca-bundle.crt && \
#     ln -s /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /etc/ssl/certs/ca-certificates.crt && \
#     mkdir -p /etc/pki/ca-trust/extracted/openssl
# COPY scripts/fix-certs .
# RUN ./fix-certs

# Removing kernel-headers seems to remove glibc and all packages which use them
# Install s/w dev tools for fitscut build
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

# # Install fitscut
# COPY scripts/install-fitscut  .
# RUN ./install-fitscut   /usr/local && \
#    rm ./install-fitscut && \
#    echo "/usr/local/lib" >> /etc/ld.so.conf && \
#    ldconfig

WORKDIR /home/developer
RUN mkdir /home/developer/mosaic-ml
ADD mosaic-ml/ /home/developer/mosaic-ml/
RUN chown -R developer:developer /home/developer

COPY requirements.txt /home/developer/.
RUN python -m pip install --upgrade pip && python -m pip install -r /home/developer/requirements.txt



# COPY ./scripts/crds_s3_get /home/developer/crds_s3_get
# RUN chmod +x /home/developer/crds_s3_get

# # CRDS cache mount point or container storage.
# RUN mkdir -p /grp/crds/cache && chown -R developer.developer /grp/crds/cache


# ------------------------------------------------
USER developer
ENV SVM_QUALITY_TESTING=on
#RUN conda init bash && source ~/.bashrc
#RUN cd caldp  &&  pip install .[dev,test]

CMD ["/bin/bash"]

