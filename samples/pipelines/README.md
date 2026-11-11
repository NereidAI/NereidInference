<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Sample pipeline definitions

These files are importable application material, not platform defaults. The
device-inference service starts with an empty pipeline registry; no
business-specific graph is compiled into the binary or shipped under
`cmd/res/`.

## Onboard a sample

Two supported channels, both domain-free mechanisms:

1. **Provision file (A channel).** Copy the sample next to the deployment's
   `res/` configuration and set `Inference.Pipelines.ProvisionFile` to its
   path, for example `res/sample-pipelines.yaml`. The file is read at
   startup; a missing file yields an empty registry rather than a startup
   failure.
2. **Manager resource (B channel).** PUT the complete definition array
   through the `PipelinesJSON` EdgeX resource. Accepted tables persist under
   `ModelRoot/runtime/pipelines/pipelines.json` and survive restarts;
   `pipelineId` + `version` is permanently immutable.

Every graph must still satisfy the closed five-node linear vocabulary
(source → infer → transform → infer → emit) validated by
`internal/pipeline.ValidateDefinition`. Adding a new application means
adding a new sample file or PUT payload — never a code change in the
platform kernel. See the Platform Kernel Closure treaty in the
NereidOverview repository.
