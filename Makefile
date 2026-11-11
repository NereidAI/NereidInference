# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Optional local build acceleration (treaty §19.3): copy local.mk.example to
# local.mk (git-ignored) and set regional proxy/mirror endpoints there. A
# missing file means a pristine direct-internet build, as used by CI.
-include local.mk

.PHONY: build test unittest lint clean image docker docker-multiarch

# change the following boolean flag to enable or disable the Full RELRO (RELocation Read Only) for linux ELF (Executable and Linkable Format) binaries
ENABLE_FULL_RELRO=true
# change the following boolean flag to enable or disable PIE for linux binaries which is needed for ASLR (Address Space Layout Randomization) on Linux, the ASLR support on Windows is enabled by default
ENABLE_PIE=true

MICROSERVICES=cmd/device-inference

.PHONY: $(MICROSERVICES)

ARCH=$(shell uname -m)

DOCKERS=docker_device_inference
.PHONY: $(DOCKERS)

# No VERSION file: the git tag is the single version source (treaty §19). On
# an exact v* tag checkout use its bare SemVer (v0.0.1 -> 0.0.1); any other
# checkout builds the rolling dev channel 0.0.0-dev. CI may inject VERSION.
VERSION ?= $(shell ref=$$(git describe --tags --exact-match 2>/dev/null) && echo "$${ref#v}" || echo 0.0.0-dev)
TAG ?= $(VERSION)
GIT_SHA=$(shell git rev-parse HEAD 2>/dev/null || echo unknown)
# Three-tag scheme (treaty §19.2): VERSION plus UTC-timestamp and short-SHA
# traceability tags. No latest tag; Make only builds, never pushes (§19.1).
GIT_SHORT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_TS := $(shell date -u +%Y%m%d-%H%M)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
SDKVERSION=$(shell cat ./go.mod | grep 'github.com/edgexfoundry/device-sdk-go/v4 v' | sed 's/require//g' | awk '{print $$2}')
REGISTRY ?= ghcr.io/nereidai
IMAGE_NAME ?= nereidinference
DOCKER_IMAGE ?= $(REGISTRY)/$(IMAGE_NAME)
DEV_DOCKER_IMAGE ?= edgexfoundry/device-inference
PLATFORMS ?= linux/amd64,linux/arm64
OCI_IMAGE ?= dist/$(IMAGE_NAME)-$(TAG).oci.tar
# Base coordinates remain upstream by default; override them when a local
# registry mirror lacks a specific tag without changing the Dockerfile.
BASE ?= golang:1.27.3-alpine3.24
RUNTIME_BASE ?= alpine:3.24.4
# Build-network endpoints (treaty §17): empty by default, injected via the
# shell environment; empty values keep pristine upstream sources.
GOPROXY ?=
ALPINE_MIRROR ?=

DOCKER_BUILD_ARGS = \
	--build-arg VERSION=$(VERSION) \
	--build-arg ADD_BUILD_TAGS=$(ADD_BUILD_TAGS) \
	--build-arg BASE=$(BASE) \
	--build-arg RUNTIME_BASE=$(RUNTIME_BASE) \
	--build-arg GOPROXY=$(GOPROXY) \
	--build-arg ALPINE_MIRROR=$(ALPINE_MIRROR) \
	--build-arg GIT_SHA=$(GIT_SHA) \
	--build-arg BUILD_DATE=$(BUILD_DATE)

IMAGE_TAGS = -t $(DOCKER_IMAGE):$(TAG) \
	-t $(DOCKER_IMAGE):$(TAG)-$(BUILD_TS) \
	-t $(DOCKER_IMAGE):$(TAG)-$(GIT_SHORT)
DEV_IMAGE_TAGS = \
	-t $(DEV_DOCKER_IMAGE):$(GIT_SHA) \
	-t $(DEV_DOCKER_IMAGE):$(TAG)-dev

ifeq ($(ENABLE_FULL_RELRO), true)
	ENABLE_FULL_RELRO_GOFLAGS = -bindnow
endif

# Cross-linking PIE binaries for musl requires the target loader path: Go
# otherwise probes the build host and can embed a glibc interpreter for arm64.
ELF_INTERPRETER ?=
ifneq ($(strip $(ELF_INTERPRETER)),)
	ELF_INTERPRETER_GOFLAGS = -I $(ELF_INTERPRETER)
endif

GOFLAGS=-ldflags "-s -w -X github.com/edgexfoundry/device-inference.Version=$(VERSION) \
                  -X github.com/edgexfoundry/device-sdk-go/v4/internal/common.SDKVersion=$(SDKVERSION) \
                  $(ENABLE_FULL_RELRO_GOFLAGS) $(ELF_INTERPRETER_GOFLAGS)" \
                   -trimpath -mod=readonly

ifeq ($(ENABLE_PIE), true)
	GOFLAGS += -buildmode=pie
endif

build: $(MICROSERVICES)

build-nats:
	make -e ADD_BUILD_TAGS=include_nats_messaging build

build-noziti:
	make -e ADD_BUILD_TAGS=no_openziti build

tidy:
	go mod tidy

cmd/device-inference:
	CGO_ENABLED=0 go build -tags "$(ADD_BUILD_TAGS)" $(GOFLAGS) -o $@ ./cmd

unittest:
	go test ./... -coverprofile=coverage.out

lint:
	@which golangci-lint >/dev/null || echo "WARNING: go linter not installed. To install, run make install-lint"
	@if [ "z${ARCH}" = "zx86_64" ] && which golangci-lint >/dev/null ; then golangci-lint run --config .golangci.yml ; else echo "WARNING: Linting skipped (not on x86_64 or linter not installed)"; fi

install-lint:
	sudo curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $$(go env GOPATH)/bin v2.15.0

test: unittest lint
	go vet ./...
	gofmt -l $$(find . -type f -name '*.go'| grep -v "/vendor/")
	[ "`gofmt -l $$(find . -type f -name '*.go'| grep -v "/vendor/")`" = "" ]
	./bin/test-attribution-txt.sh

clean:
	rm -f $(MICROSERVICES) $(OCI_IMAGE)

# Keep the native image in Docker for Compose, and export one OCI archive with
# a manifest index for every requested target. The classic Docker image store
# cannot load a multi-platform index; publishing that index is a release step.
docker: $(DOCKERS) image

docker_device_inference:
	docker build $(DOCKER_BUILD_ARGS) $(DEV_IMAGE_TAGS) .

image:
	mkdir -p $(dir $(OCI_IMAGE))
	rm -f $(OCI_IMAGE)
	docker buildx build \
		--platform $(PLATFORMS) \
		$(DOCKER_BUILD_ARGS) \
		$(IMAGE_TAGS) \
		--output type=oci,dest=$(OCI_IMAGE) \
		.
	./bin/test-oci-image.sh $(OCI_IMAGE) $(PLATFORMS)

# Compatibility with the initial multi-arch development target.
docker-multiarch: image

docker-nats:
	make -e ADD_BUILD_TAGS=include_nats_messaging docker

docker-noziti:
	make -e ADD_BUILD_TAGS=no_openziti docker

vendor:
	CGO_ENABLED=0 go mod vendor
