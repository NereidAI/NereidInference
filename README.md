<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# device-inference

[![CI](https://github.com/NereidAI/NereidInference/actions/workflows/ci.yml/badge.svg)](https://github.com/NereidAI/NereidInference/actions/workflows/ci.yml)
[![Dev Image](https://github.com/NereidAI/NereidInference/actions/workflows/publish-dev.yml/badge.svg)](https://github.com/NereidAI/NereidInference/actions/workflows/publish-dev.yml)
[![Release](https://github.com/NereidAI/NereidInference/actions/workflows/release.yml/badge.svg)](https://github.com/NereidAI/NereidInference/actions/workflows/release.yml)
[![DCO](https://github.com/NereidAI/NereidInference/actions/workflows/dco.yml/badge.svg)](https://github.com/NereidAI/NereidInference/actions/workflows/dco.yml)
[![License](https://img.shields.io/github/license/NereidAI/NereidInference)](LICENSE)
[![Architectures](https://img.shields.io/badge/arch-amd64%20%7C%20arm64-627185?logo=linux&logoColor=white)](https://github.com/NereidAI/NereidInference/actions/workflows/release.yml)
[![Go](https://img.shields.io/badge/go-1.27.3-00ADD8?logo=go&logoColor=white)](go.mod)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe519e?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)
[![GHCR nereidinference](https://img.shields.io/badge/GHCR-nereidinference-2496ED?logo=docker&logoColor=white)](https://github.com/orgs/NereidAI/packages/container/package/nereidinference)
[![EdgeX Foundry](https://img.shields.io/badge/EdgeX-4.0-blue.svg)](https://www.edgexfoundry.org/)

EdgeX device service for generic, descriptor-driven media inference. One
service instance per node runs any number of inference **jobs** (one per
provisioned device); each job acquires frames from a camera or image source,
calls a standards-based model server over KServe gRPC v2 or an OpenAI
compatible HTTP API, decodes the raw tensors according to the model package
descriptor, and publishes normalized structured events to the EdgeX message
bus, plus an optional local annotated MJPEG preview.

The service is model and silicon vendor neutral: it contains no vendor SDK,
no hard-coded tensor layouts and no business logic. Classification,
detection, segmentation, pose/keypoint, embedding, dense-map, image-to-image
and CTC-sequence models differ only in their package descriptor and the
registered decoder kind; model servers (OpenVINO Model Server, Triton, vLLM,
MindIE, ...) are deployed independently.

> Status: the generic pipeline, model catalog, KServe/OpenAI backends and the
> first decoder set are implemented on EdgeX 4.0.2. Live validation has passed
> for the HTTP JPEG → OVMS KServe gRPC → decoder → EdgeX event/preview path,
> one real RTSP H.264 camera, one ONVIF Snapshot camera, and managed Triton
> with ONNX Runtime on CPU. Default ONVIF `GetStreamUri`, H.265, reconnect and
> credential rotation, production Horizon packages, GPU/OpenVINO-backend
> Triton, and other-vendor acceptance remain open.

## Architecture in one picture

```
camera RTSP ──► ffmpeg child process (H.264/H.265 decode, rawvideo pipe)
MJPEG/JPEG ──► pure-Go frame source          ──► preprocess ──► KServe gRPC / OpenAI
                                                                       │
EdgeX core ◄── structured events (async) ◄── generic decoder ◄────────┘
operators  ◄── annotated MJPEG preview (local, optional)
model packages (.tar.gz with nereid-model.json) ──► inbox / ESS watcher ──► active model root
```

Key properties:

- **One ffmpeg per source URL**, reference counted and fanned out to N jobs;
  each job samples at its own FPS with at most one in-flight inference and
  drop-oldest semantics, so a slow model only lowers that job's frame rate —
  latency stays bounded and nothing queues up.
- **No CGo**: the Go binary is statically built (`CGO_ENABLED=0`); ffmpeg is a
  runtime Alpine package, executed as a credential-dropped child process.
- **Local model acquisition**: the service watches an Open Horizon ESS local
  object directory and/or a plain inbox directory, validates and unpacks
  model packages into an `active/` model root shared read-only with the model
  server.
- **Credentials are federated, not typed per job**: for ONVIF cameras the
  service resolves the stream URI and secret name through EdgeX
  core-metadata/core-command/secret-store data already registered by
  device-onvif-camera.

## Building the service

```bash
make build
```

The binary is produced at `cmd/device-inference`.

Other targets:

```bash
make test        # unit tests, golangci-lint, go vet, gofmt, Attribution.txt gate
make image       # canonical linux/amd64,linux/arm64 OCI archive
make docker      # native development Docker tags plus the OCI archive
make tidy        # go mod tidy (then extend Attribution.txt)
```

There is no `VERSION` file: the git tag is the single version source
(treaty §19). An exact `v*` tag checkout builds its bare SemVer
(`v0.0.1` -> `0.0.1`); every other checkout builds the rolling dev channel
`0.0.0-dev`, and CI may inject `VERSION` explicitly as a build argument. The
Dockerfile follows the EdgeX multi-stage convention; `BASE`, `RUNTIME_BASE`
and `GOPROXY` are overridable for restricted networks. `make image` exports
`dist/nereidinference-<version>.oci.tar` carrying the three-tag scheme
(treaty §19.2): `ghcr.io/nereidai/nereidinference:<version>` plus
`:<version>-<utc-timestamp>` and `:<version>-<git-short-sha>`, never `latest`.
Its manifest index covers `linux/amd64` and `linux/arm64`; exporting it never
pushes a registry. For regional mirrors, put the endpoints in the
git-ignored `local.mk` (copy `local.mk.example`, treaty §19.3) or inject them
as environment variables (root treaty §17; empty values keep the pristine
upstream sources):

```bash
make image \
  PLATFORMS=linux/amd64,linux/arm64 \
  BASE=golang:1.27.3-alpine3.24 \
  RUNTIME_BASE=alpine:3.24.4 \
  GOPROXY=<go-module-proxy> \
  ALPINE_MIRROR=<apk-mirror-url>
```

`make docker` additionally loads the native
`edgexfoundry/device-inference:<version>-dev` and `:<git-sha>` development tags
for local runbooks and Compose overrides. The Go binary is cross-compiled for each target. Both runtime
images include ffmpeg/ffprobe; the Intel iHD VA-API driver is installed only on
amd64 because Alpine does not publish `intel-media-driver` for arm64. Test an
arm64 image on arm64 hardware before claiming hardware or camera compatibility.

### Image CI and publication

`.github/workflows/ci.yml` runs the repository test gate, builds the same OCI
archive, verifies its platform index, and uploads it as a workflow artifact.
Pull requests and branch builds never authenticate to or push a registry.

`.github/workflows/release.yml` publishes the multi-arch image to GHCR only
when a pushed `v*` tag triggers it; the organization reusable workflow at
`@v1` derives the bare SemVer `VERSION` from the tag itself, so no VERSION
file is involved. `.github/workflows/publish-dev.yml` republishes the rolling
`0.0.0-dev` channel — plus an immutable `0.0.0-dev-<sha>` tag — on every push
to `main`; pull requests never publish. Manual dispatch is not supported.
These are the only publication paths (treaty §19.1): the Makefile has no push
target. Each release run publishes the three tags — `VERSION`,
`VERSION-<utc-timestamp>` and `VERSION-<git-short-sha>` — with no `latest`.
Copying a locally built image to a private/air-gapped hub is the operator's
own decision, done outside the repository with a manual `docker tag`/
`docker push` or NereidOverview's `harbor-sync.sh`.

## Running locally

Bring up the EdgeX 4.0 core stack from
[edgex-compose](https://github.com/edgexfoundry/edgex-compose), then:

```bash
docker compose up -d device-inference
```

The local service sets `pull_policy: build`, so this ordinary `up` command
builds `device-inference` from the current working tree before starting it;
no separate `make docker` or `--build` flag is required. Docker's build cache
still avoids rebuilding unchanged layers.

`docker-compose.yaml` in this repository is a **developer overlay** onto the
official compose stack (external `edgex_edgex-network`). The bundled OVMS
container starts from an empty DS-generated multi-model configuration; the
service adds or removes one OVMS servable for every active inference binding
and target device. Production deployment uses Open Horizon service definitions.

## Configuration

Service configuration follows the standard EdgeX layout in
[`cmd/res/configuration.yaml`](cmd/res/configuration.yaml).
Device profiles shipped in [`cmd/res/profiles/`](cmd/res/profiles/) define the
reading resources per task family. The singleton `nereid-inference` manager
device exposes model inventory; media inference jobs are not provisioned by
default.

A job is an EdgeX device whose `Inference` protocol properties select the
source, model, schedule and output options:

```yaml
protocols:
  Inference:
    SourceKind: onvif-camera      # onvif-camera | onvif-snapshot | rtsp | mpjpeg | jpeg | none
    SourceDevice: camera-01       # EdgeX name used for URI/secret/snapshot federation
    StreamProfile: Profile_2     # optional ONVIF Media ProfileToken (main/sub), onvif-camera only
    SecretName: onvif-camera-01  # optional, resolved from camera metadata when omitted
    FFmpegExtra: "-fflags nobuffer -flags low_delay"  # optional allowlist argv, ffmpeg sources only
    ModelID: yiqisoft-yolov8n
    ModelVersion: "1.0.0"
    TargetDevice: CPU             # OVMS binding target: CPU | GPU | GPU.0 | NPU | ...
    Endpoint: edgex-ovms:9000     # optional, defaults to service config
    Trigger: continuous           # continuous | periodic | event | command
    FPS: "5"
```

For the minimal YuNet → five-point alignment → SFace chain, use profile
`nereid-inference-pipeline` and replace `ModelID`/`ModelVersion` with the exact
property `PipelineId: face-recognition-v1`. The registry starts empty: that
graph is application material shipped as the opt-in sample
`samples/pipelines/face-recognition-v1.yaml`, onboarded through a provision
file or a `PipelinesJSON` PUT — the binary ships no business pipeline. The
manager resource `PipelinesJSON` exposes the immutable, versioned registry.
See the [face pipeline provisioning and descriptor guide](docs/2026-11-06-face-pipeline-provisioning.md)
for API calls and the synthetic-only validation status; real NF face-package
validation is still open.

### Preview security

The preview port is an unauthenticated operator diagnostic, not a business
API. In containers it defaults to `0.0.0.0:18080` so published ports work;
restrict it with Docker/Compose host bindings, network policies or a reverse
proxy. Set `Preview.Bind` to `127.0.0.1:18080` for a host-local-only service.

## Decoder families

The package descriptor selects a tensor-shape decoder, never a vendor or task
fork. Built-in kinds are:

- `detection.anchors` (YOLO-style `[1,4+nc,A]`), `detection.rows` and
  `detection.yunet` (descriptor-bound cls/obj/bbox/kps at strides 8/16/32)
- `classification.probs` and `classification.multilabel`
- `regression.values`
- `segmentation.semantic` and `segmentation.instance`
- `keypoints`
- `embeddings.vector`
- `dense.map`
- `image.tensor`
- `sequence.ctc`
- `composite`
- `passthrough.text` for OpenAI-compatible chat/VLM responses

## Packaging and contribution

- The distributable unit is one gzip tar package (`*.tar.gz`) with a root
  `nereid-model.json`, checksummed payloads and optional labels. NereidForge
  owns model conversion and all model-specific parameters; NereidHorizon only
  transports the package; this service unpacks and follows the descriptor and
  never generates, repairs or guesses a model contract.
- Artifact format and inference runtime are orthogonal: current packages use
  `artifact.format: onnx` or `openvino-ir`; a future Triton/TensorRT package may
  declare additional formats and `runtime.server.family: triton`. The catalog
  accepts the standard package, while the selected runtime adapter rejects an
  artifact it cannot serve. Deployment-only selectors such as `TargetDevice`
  arrive in the NL inference binding, not the package.
- For OVMS packages, the descriptor declares `runtime.server.family: ovms` and
  the server-ready repository location in `runtime.repository_path` containing
  either `<version>/model.onnx` or `<version>/model.xml` plus `model.bin`. The
  numeric repository directory remains OVMS package-local versioning (normally
  `1`) and is distinct from the Nereid semantic model version.
- `LICENSE`: standard Apache-2.0 text.
- `Attribution.txt` contains exact-version license and notice texts for every
  module in `go.mod`, package-local overrides and the Go standard library;
  `bin/test-attribution-txt.sh` checks their presence and versions in `make test`.
- Code follows EdgeX Foundry style: English comments, SPDX headers,
  `golangci-lint` configuration identical to other v4 device services.

## License

NereidInference's own code is licensed under Apache-2.0; see `LICENSE` for the
standard license text. The project is based in part on contributions from
other companies and open-source projects, including EdgeX Foundry and Kubeflow
KServe. See `NOTICE` for direct source provenance and third-party attribution;
third-party components retain their respective licenses.

## Acknowledgements

NereidInference builds on contributions from the
[EdgeX Foundry](https://www.edgexfoundry.org/) community. We thank EdgeX
Foundry and all of its contributors for their work on the device-service
ecosystem and [device-sdk-go](https://github.com/edgexfoundry/device-sdk-go).
NereidInference is an independent work, not part of, or endorsed by, EdgeX
Foundry.

We also thank the [Kubeflow KServe](https://github.com/kserve/kserve)
community for the gRPC v2 inference protocol and generated protocol code.
NereidInference is an independent work, not part of, or endorsed by, Kubeflow
KServe.
