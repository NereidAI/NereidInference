// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelcatalog

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// maxPayloadFile bounds one unpacked model package file. Model weights are
// consumed from the active tree by the inference server; the DS itself only
// needs descriptor, labels and metadata, so this rejects tar bombs.
const (
	maxPayloadFile  = 16 * 1024 * 1024 * 1024
	maxPayloadTotal = 4 * maxPayloadFile
	maxPayloadFiles = 100_000
)

// extractPackage opens the required gzip-compressed tar package and unpacks it
// into dst. It returns false for non-gzip/tar ESS objects and gzip tars that do
// not contain a root nereid-model.json; malformed or unsafe archives error.
func extractPackage(path, dst string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		// ESS may carry unrelated business objects; only gzip packages are
		// candidates for the standardized Nereid model package format.
		return false, nil
	}
	defer gr.Close()
	tr := tar.NewReader(gr)

	foundDescriptor := false
	count := 0
	var total int64
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if count == 0 {
				return false, nil
			}
			return false, err
		}
		count++
		if count > maxPayloadFiles {
			return false, fmt.Errorf("archive exceeds %d members", maxPayloadFiles)
		}
		clean := filepath.Clean(hdr.Name)
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return false, fmt.Errorf("unsafe archive member %q", hdr.Name)
		}
		target := filepath.Join(dst, clean)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return false, err
			}
		case tar.TypeReg:
			remaining := maxPayloadTotal - total
			if hdr.Size < 0 || hdr.Size > maxPayloadFile || hdr.Size > remaining || remaining <= 0 {
				return false, fmt.Errorf("archive member %q exceeds payload budget", hdr.Name)
			}
			limit := int64(maxPayloadFile)
			if remaining < limit {
				limit = remaining
			}
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return false, err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
			if err != nil {
				return false, err
			}
			n, err := io.CopyN(out, tr, limit+1)
			out.Close()
			total += n
			if err != nil && err != io.EOF {
				return false, err
			}
			if n > limit || total > maxPayloadTotal {
				return false, fmt.Errorf("payload member %q exceeds extraction budget", hdr.Name)
			}
		}
		if clean == "nereid-model.json" {
			foundDescriptor = true
		}
	}
	return foundDescriptor, nil
}
