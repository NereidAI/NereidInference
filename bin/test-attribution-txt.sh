#!/bin/bash -e

# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Verify that every locked module has an exact-version attribution entry and
# that package-local and Go toolchain notices cannot silently disappear.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

EXIT_CODE=0

cd "$GIT_ROOT"

if [ ! -f Attribution.txt ]; then
    echo "An Attribution.txt file is missing, please add"
    exit 1
fi

while read -r lib version; do
    case "$version" in
        v*) ;;
        *) continue ;;
    esac
    if ! grep -Fq "Module: ${lib}" Attribution.txt; then
        echo "An attribution for ${lib} is missing from Attribution.txt, please add"
        ((EXIT_CODE = 1))
    elif ! grep -A1 -F "Module: ${lib}" Attribution.txt | grep -Fq "Version: ${version}"; then
        echo "The attribution for ${lib} does not name locked version ${version}"
        ((EXIT_CODE = 1))
    fi
done < <(cat "$GIT_ROOT/go.mod" | grep -v 'module ' | grep -v TODO | grep '/' | sed 's/require //' | sed 's/replace //' | awk '{print $1, $2}')

if grep -Eq '^License: .*(Unspecified|MIT or BSD-style)' Attribution.txt; then
    echo "Attribution.txt contains an ambiguous license identifier"
    ((EXIT_CODE = 1))
fi

while read -r required; do
    if ! grep -Fq "$required" Attribution.txt; then
        echo "A required package-local or toolchain attribution is missing: ${required}"
        ((EXIT_CODE = 1))
    fi
done <<'EOF'
Module: github.com/go-jose/go-jose/v4/json
Module: github.com/openziti/foundation/v2/uuidz
Module: github.com/go-openapi/runtime/middleware/denco
Module: Go standard library
Module: Go standard library vendored golang.org/x/crypto
Module: Go standard library vendored golang.org/x/net
Module: Go standard library vendored golang.org/x/sys
Module: Go standard library vendored golang.org/x/text
EOF

exit $EXIT_CODE
