// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelcatalog

import (
	"strconv"
	"strings"
	"time"
)

// ledger is the on-disk audit trail of package ingestion attempts.
type ledger struct {
	Records []ledgerRecord `json:"records"`
}

// ledgerRecord is one package version entry in the audit ledger.
type ledgerRecord struct {
	ObjectID    string    `json:"object_id"`
	ID          string    `json:"id"`
	Version     string    `json:"version"`
	Revision    string    `json:"revision"`
	State       string    `json:"state"`
	SourcePath  string    `json:"source_path"`
	SourceSize  int64     `json:"source_size,omitempty"`
	SourceModAt time.Time `json:"source_modified_at,omitempty"`
	InstalledAt time.Time `json:"installed_at"`
}

// compareRevision orders revisions numerically when both sides are integers
// (NF uses the numeric optimization job id), lexically otherwise.
func compareRevision(a, b string) int {
	an, aerr := strconv.Atoi(a)
	bn, berr := strconv.Atoi(b)
	if aerr == nil && berr == nil {
		switch {
		case an > bn:
			return 1
		case an < bn:
			return -1
		default:
			return 0
		}
	}
	return strings.Compare(a, b)
}

// compareVersions implements dotted numeric version ordering with lexical
// fallback; it is intentionally dependency-free.
func compareVersions(a, b string) int {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	n := len(pa)
	if len(pb) > n {
		n = len(pb)
	}
	for i := 0; i < n; i++ {
		var av, bv string
		if i < len(pa) {
			av = pa[i]
		}
		if i < len(pb) {
			bv = pb[i]
		}
		an, ae := strconv.Atoi(av)
		bn, be := strconv.Atoi(bv)
		switch {
		case ae == nil && be == nil && an != bn:
			if an > bn {
				return 1
			}
			return -1
		case av != bv:
			return strings.Compare(av, bv)
		}
	}
	return 0
}
