#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0
#
# Verify that an exported OCI archive contains exactly the requested runnable
# platforms. Buildx provenance attestations use unknown/unknown and are ignored.

set -euo pipefail

archive=${1:?"usage: $0 <oci-archive> <comma-separated-platforms>"}
platforms=${2:?"usage: $0 <oci-archive> <comma-separated-platforms>"}

if [ ! -s "${archive}" ]; then
    echo "OCI archive is missing or empty: ${archive}" >&2
    exit 1
fi

root_index=$(tar -xOf "${archive}" index.json)
mapfile -t index_digests < <(
    jq -r '.manifests[] | select(.mediaType == "application/vnd.oci.image.index.v1+json") | .digest' \
        <<<"${root_index}" | sort -u
)
if [ "${#index_digests[@]}" -ne 1 ]; then
    echo "expected one unique multi-platform image index, found ${#index_digests[@]}" >&2
    exit 1
fi

digest=${index_digests[0]#sha256:}
image_index=$(tar -xOf "${archive}" "blobs/sha256/${digest}")
actual=$(
    jq -r '.manifests[] | select(.platform.os == "linux") |
        [.platform.os, .platform.architecture, (.platform.variant // "")] |
        join("/") | sub("/$"; "")' <<<"${image_index}" | sort -u
)
expected=$(printf '%s' "${platforms}" | tr ',' '\n' | sed '/^$/d' | sort -u)

if [ -z "${expected}" ]; then
    echo "at least one target platform is required" >&2
    exit 1
fi
if ! diff -u <(printf '%s\n' "${expected}") <(printf '%s\n' "${actual}"); then
    echo "OCI image platforms do not match the requested targets" >&2
    exit 1
fi

# Check the final config for every runnable platform, not just the index. This
# prevents a locally inspected image from diverging from the CI OCI artifact.
mapfile -t manifest_digests < <(
    jq -r '.manifests[] | select(.platform.os == "linux") | .digest' \
        <<<"${image_index}" | sort -u
)
required_labels=(
    license
    copyright
    org.opencontainers.image.title
    org.opencontainers.image.description
    org.opencontainers.image.vendor
    org.opencontainers.image.licenses
    org.opencontainers.image.version
    org.opencontainers.image.revision
    org.opencontainers.image.created
    org.opencontainers.image.source
    org.opencontainers.image.url
    org.opencontainers.image.documentation
    org.opencontainers.image.authors
)

for manifest_digest in "${manifest_digests[@]}"; do
    manifest=$(tar -xOf "${archive}" "blobs/sha256/${manifest_digest#sha256:}")
    config_digest=$(jq -r '.config.digest' <<<"${manifest}")
    config=$(tar -xOf "${archive}" "blobs/sha256/${config_digest#sha256:}")

    if ! jq -e '.config.Labels as $labels | ($labels | type) == "object" and ($labels | length) == 13' \
        <<<"${config}" >/dev/null; then
        echo "OCI image config must contain exactly 13 labels: ${manifest_digest}" >&2
        exit 1
    fi
    for label in "${required_labels[@]}"; do
        if ! jq -e --arg label "${label}" '.config.Labels | has($label)' \
            <<<"${config}" >/dev/null; then
            echo "OCI image config is missing label ${label}: ${manifest_digest}" >&2
            exit 1
        fi
    done
    if ! jq -e '
        .config.Labels as $labels |
        $labels["license"] == "SPDX-License-Identifier: Apache-2.0" and
        $labels["copyright"] == "Copyright (c) 2026: YIQISOFT" and
        $labels["org.opencontainers.image.title"] == "nereidinference" and
        $labels["org.opencontainers.image.description"] == "Edge inference runtime — EdgeX device service: camera frames in, KServe gRPC v2/OpenAI model-server calls, decoded tensors out as EdgeX events" and
        $labels["org.opencontainers.image.vendor"] == "YIQISOFT" and
        $labels["org.opencontainers.image.licenses"] == "Apache-2.0" and
        $labels["org.opencontainers.image.source"] == "https://github.com/NereidAI/NereidInference" and
        $labels["org.opencontainers.image.url"] == "https://github.com/NereidAI/NereidInference" and
        $labels["org.opencontainers.image.documentation"] == "https://github.com/NereidAI/NereidInference#readme" and
        $labels["org.opencontainers.image.authors"] == "NereidAI (https://github.com/NereidAI)" and
        ($labels["org.opencontainers.image.version"] | type == "string" and length > 0) and
        ($labels["org.opencontainers.image.revision"] | type == "string" and length > 0) and
        $labels["org.opencontainers.image.revision"] != "unknown" and
        ($labels["org.opencontainers.image.created"] | test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"))
    ' <<<"${config}" >/dev/null; then
        echo "OCI image config has invalid legal or metadata labels: ${manifest_digest}" >&2
        exit 1
    fi
done

printf 'OCI image %s contains platforms and valid labels:\n%s\n' "${archive}" "${actual}"
