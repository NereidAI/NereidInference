<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Contributing to NereidInference

Thank you for contributing to NereidInference. This repository follows EdgeX
Foundry device-service conventions and the NereidAI engineering treaty.

## Development requirements

Use the Go toolchain and runtime versions declared by the repository. Keep all
code, comments, documentation, commit messages, and API messages in English.
Do not add credentials, internal hostnames, private network addresses, or
internal registry endpoints to the public tree.

Run the relevant checks before submitting a change:

```bash
make unittest
make lint
make test
make build
bin/test-attribution-txt.sh
```

`make test` runs the unit tests, lint, `go vet`, formatting checks, and the
Attribution.txt gate. The following image targets build locally and do not
push to a registry:

```bash
make image
make docker-multiarch
```

## Developer Certificate of Origin

Every commit must include a Developer Certificate of Origin sign-off matching
its author. Create commits with:

```bash
git commit -s
```

The DCO workflow checks every non-merge commit in a pull request. Do not use a
sign-off belonging to another author.

## SPDX and copyright headers

Handwritten source, configuration, workflow, script, and Markdown files must
carry the project copyright and `SPDX-License-Identifier: Apache-2.0` header
using the file's comment syntax. Keep a shell shebang on the first line and
place its header immediately after the shebang. Markdown files use a single
HTML comment on the first line.

The approved exceptions are:

- `LICENSE`, `NOTICE`, and `Attribution.txt`, which are legal texts;
- JSON files, `go.mod`, and `go.sum`, which do not accept the required headers;
- `internal/infer/kserve/grpcv2/grpc_predict_v2*.pb.go`, whose upstream
  Kubeflow generated-code headers must remain unchanged;
- `third_party/`, when present, whose upstream notices govern the files; and
- `AGENTS.md`, whose public-tree disposition is an owner decision.

Do not add a project SPDX header to `Attribution.txt` or alter the Kubeflow
generated-code headers.

## EdgeX style and third-party attribution

Follow EdgeX Foundry code style, including English comments, Go formatting,
and the repository's `golangci-lint` configuration. Preserve upstream copyright
and license notices when adapting source. Record direct source provenance in
`NOTICE` and keep module-level license texts in `Attribution.txt`.

Whenever `go.mod` or `go.sum` changes, update `Attribution.txt` with the exact
locked module versions and any applicable package-local notices. Run
`bin/test-attribution-txt.sh` and `make test` after the update. Do not use
ambiguous license labels or unpinned source links.

## Cross-repository identifiers

Model IDs, pipeline IDs, service keys, protocol properties, and other
cross-repository identifiers are shared contracts. Do not rename them
unilaterally. Coordinate any required change with the owning NereidAI
repositories and update all affected contracts together.

There is no `VERSION` file. The git tag is the version source for releases;
do not introduce a competing version file or push images from local build
targets.
