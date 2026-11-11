# Copyright (c) 2023 Intel
# Copyright (c) 2024 IOTech Ltd
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0
#
# Based on EdgeX device-sdk-go example/cmd/device-simple/Dockerfile and
# modified for NereidInference.
#
# EdgeX style multi-stage build for device-inference. The runtime image
# additionally carries ffmpeg/ffprobe and the iHD VA-API driver: the frame
# source manager shells out to ffmpeg for RTSP transport and H.264/H.265
# decoding (rawvideo over a pipe), optionally hardware-accelerated on a
# mapped DRM render node; the Go binary itself is statically built with
# CGO_ENABLED=0.

# BASE/RUNTIME_BASE are overridable for private registries that mirror only a
# subset of tags; both defaults are pinned for reproducible builds.
ARG BASE=golang:1.27.3-alpine3.24
ARG RUNTIME_BASE=alpine:3.24.4
FROM --platform=${BUILDPLATFORM} ${BASE} AS builder

ARG TARGETOS
ARG TARGETARCH
ARG ALPINE_PKG_BASE="make git openssh-client"
ARG ALPINE_PKG_EXTRA=""
ARG ADD_BUILD_TAGS=""
# Go module proxy for air-gapped or regional build networks (treaty §17);
# empty makes the toolchain use its pristine upstream default proxy.
ARG GOPROXY=""
# Optional Alpine package mirror URL without a trailing slash. Empty keeps
# the upstream dl-cdn.alpinelinux.org repository.
ARG ALPINE_MIRROR=""

# set the working directory
WORKDIR /device-inference
ENV GOPROXY=${GOPROXY}

# Install our build time packages.
RUN if [ -n "${ALPINE_MIRROR}" ]; then \
      sed -i "s|https*://dl-cdn.alpinelinux.org|${ALPINE_MIRROR}|g" /etc/apk/repositories; \
    fi \
 && apk add --update --no-cache ${ALPINE_PKG_BASE} ${ALPINE_PKG_EXTRA}

COPY go.mod go.sum vendor* ./
RUN [ ! -d "vendor" ] && go mod download all || echo "skipping..."

COPY . .
# To run tests in the build container:
#   docker build --build-arg 'MAKE=build test' .
# Default is the rolling dev channel; the release workflow injects bare
# SemVer (0.0.1) derived from the v-prefixed git tag. "make -e" lets this
# environment value override the Makefile's tag-sourced VERSION derivation,
# which itself falls back to 0.0.0-dev because .git is not in the build
# context (.dockerignore).
ARG VERSION=0.0.0-dev
ARG MAKE="make -e ADD_BUILD_TAGS=$ADD_BUILD_TAGS build"
# Build with the native Go toolchain and cross-compile the static target binary;
# only the runtime stage needs the target architecture. Pin the musl interpreter
# explicitly because Go cannot discover another architecture's loader on the
# build host and would otherwise embed the glibc path for an arm64 PIE binary.
RUN GOOS=${TARGETOS:-$(go env GOOS)} GOARCH=${TARGETARCH:-$(go env GOARCH)} VERSION=${VERSION} $MAKE \
    ELF_INTERPRETER=$(case "${TARGETARCH}" in \
      amd64) printf '%s' /lib/ld-musl-x86_64.so.1 ;; \
      arm64) printf '%s' /lib/ld-musl-aarch64.so.1 ;; \
    esac)

FROM ${RUNTIME_BASE}

ARG TARGETARCH
ARG ALPINE_MIRROR=""
ARG VERSION=0.0.0-dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=""

LABEL license='SPDX-License-Identifier: Apache-2.0' \
  copyright='Copyright (c) 2026: YIQISOFT' \
  org.opencontainers.image.title='nereidinference' \
  org.opencontainers.image.description='Edge inference runtime — EdgeX device service: camera frames in, KServe gRPC v2/OpenAI model-server calls, decoded tensors out as EdgeX events' \
  org.opencontainers.image.vendor='YIQISOFT' \
  org.opencontainers.image.licenses='Apache-2.0' \
  org.opencontainers.image.version="${VERSION}" \
  org.opencontainers.image.revision="${GIT_SHA}" \
  org.opencontainers.image.created="${BUILD_DATE}" \
  org.opencontainers.image.source='https://github.com/NereidAI/NereidInference' \
  org.opencontainers.image.url='https://github.com/NereidAI/NereidInference' \
  org.opencontainers.image.documentation='https://github.com/NereidAI/NereidInference#readme' \
  org.opencontainers.image.authors='NereidAI (https://github.com/NereidAI)'

# dumb-init is the EdgeX standard entrypoint wrapper; ffmpeg+ffprobe are the
# portable runtime media dependencies. Alpine publishes intel-media-driver only
# for x86 targets, so arm64 intentionally has no Intel iHD VA-API driver. The
# render node itself is mapped into the container by the deployment. Upgrade
# installed packages to avoid CVEs.
RUN if [ -n "${ALPINE_MIRROR}" ]; then \
      sed -i "s|https*://dl-cdn.alpinelinux.org|${ALPINE_MIRROR}|g" /etc/apk/repositories; \
    fi \
 && apk add --update --no-cache dumb-init ffmpeg \
 && case "${TARGETARCH:-$(uname -m)}" in \
      amd64|x86_64) apk add --no-cache intel-media-driver ;; \
      arm64|aarch64) ;; \
      *) echo "unsupported runtime architecture: ${TARGETARCH:-$(uname -m)}" >&2; exit 1 ;; \
    esac \
 && apk --no-cache upgrade

WORKDIR /
COPY --from=builder /device-inference/Attribution.txt /licenses/
COPY --from=builder /device-inference/LICENSE /licenses/
COPY --from=builder /device-inference/NOTICE /licenses/
COPY --from=builder /device-inference/cmd /

# 59992 is the provisional service port (pending EdgeX TSC assignment);
# 18080 is the local MJPEG preview server, exposed only when needed.
EXPOSE 59992 18080

ENTRYPOINT ["/device-inference"]
CMD ["-cp=keeper.http://edgex-core-keeper:59890", "--registry"]
