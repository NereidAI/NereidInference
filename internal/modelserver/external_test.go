// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

type fakeExternalProber struct {
	err   error
	calls int
}

func (p *fakeExternalProber) Probe(_ context.Context, _, _, _, _ string) error {
	p.calls++
	return p.err
}

func validRecord(id string) ServableRecord {
	return ServableRecord{
		ServableID: id,
		Family:     FamilyExternal,
		Protocol:   externalProtocolKServeGRPC,
		Endpoint:   "triton-dc-01:8001",
		ModelName:  "face_embedding",
		Descriptor: []byte(`{
			"runtime": {"backend": "kserve", "model_name": "face_embedding",
			            "server": {"family": "external"}},
			"io": {"inputs": [{"name": "images", "shape": [1,3,112,112], "layout": "NCHW", "dtype": "FP32"}],
			       "outputs": [{"name": "emb", "shape": [1,512], "layout": "NC", "dtype": "FP32"}]},
			"decoder": {"kind": "embedding"}
		}`),
	}
}

func writeExternalRuntimeRecords(t *testing.T, root string, records []ServableRecord) string {
	t.Helper()
	raw, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	raw = append(raw, '\n')
	dir := filepath.Join(root, filepath.FromSlash(externalDirRelPath))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, externalStoreFile)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestValidateServableRecord(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(*ServableRecord)
		wantErr bool
	}{
		{"valid", func(_ *ServableRecord) {}, false},
		{"wrong family", func(r *ServableRecord) { r.Family = "triton" }, true},
		{"bad protocol", func(r *ServableRecord) { r.Protocol = "http" }, true},
		{"scheme in endpoint", func(r *ServableRecord) { r.Endpoint = "http://host:8001" }, true},
		{"missing port", func(r *ServableRecord) { r.Endpoint = "host" }, true},
		{"unsafe model name", func(r *ServableRecord) { r.ModelName = "../evil" }, true},
		{"unsafe id", func(r *ServableRecord) { r.ServableID = "bad id!" }, true},
		{"bad capability device", func(r *ServableRecord) { r.Capabilities.Devices = []string{"GPU;rm"} }, true},
		{"bad descriptor backend", func(r *ServableRecord) {
			r.Descriptor = []byte(`{"runtime":{"backend":"mystery","model_name":"x"},"decoder":{"kind":"embedding"}}`)
		}, true},
		{"descriptor missing decoder", func(r *ServableRecord) {
			r.Descriptor = []byte(`{"runtime":{"backend":"kserve","model_name":"x"},"io":{"inputs":[{"name":"i","shape":[1,3,4,4],"layout":"NCHW","dtype":"FP32"}]}}`)
		}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := validRecord("datacenter-face")
			tc.mutate(&rec)
			err := ValidateServableRecord(rec)
			if tc.wantErr && err == nil {
				t.Fatal("want validation error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected validation error: %v", err)
			}
		})
	}
}

func TestExternalRegistryProvisionRuntimeMergeAndProbe(t *testing.T) {
	root := t.TempDir()
	provision := filepath.Join(t.TempDir(), "external-servables.yaml")
	if err := os.WriteFile(provision, []byte(`servables:
  - servableId: provisioned-one
    family: external
    protocol: kserve-grpc
    endpoint: dc:8001
    modelName: model_a
`), 0o600); err != nil {
		t.Fatal(err)
	}

	prober := &fakeExternalProber{}
	registry, err := NewExternal(ExternalConfig{
		ModelRoot:     root,
		ProvisionFile: provision,
		Prober:        prober,
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := registry.Lookup("provisioned-one"); !ok {
		t.Fatal("provisioned record must be visible after construction")
	}
	if got := registry.Snapshot(); len(got) != 1 || got[0].Channel != "provisioned" || got[0].State != "VALIDATING" {
		t.Fatalf("unexpected initial snapshot: %+v", got)
	}

	// Three successful probes stabilize to READY (inventory flap guard).
	for i := 0; i < externalStableCount; i++ {
		registry.scan(context.Background())
	}
	got := registry.Snapshot()
	if len(got) != 1 || got[0].State != "READY" {
		t.Fatalf("want READY after stable probes, got %+v", got)
	}
	if prober.calls < externalStableCount {
		t.Fatalf("prober calls = %d", prober.calls)
	}

	// Preseed a legacy runtime record as deployment upgrade material, then
	// construct a new registry to prove restart visibility without a writer.
	writeExternalRuntimeRecords(t, root, []ServableRecord{validRecord("runtime-one")})
	reloaded, err := NewExternal(ExternalConfig{ModelRoot: root, ProvisionFile: provision, Prober: prober})
	if err != nil {
		t.Fatal(err)
	}
	snapshot := reloaded.Snapshot()
	if len(snapshot) != 2 {
		t.Fatalf("want merged provisioned+runtime table, got %d entries", len(snapshot))
	}
	runtimeRec, ok := reloaded.Lookup("runtime-one")
	if !ok || runtimeRec.Endpoint != "triton-dc-01:8001" {
		t.Fatalf("runtime record missing: %+v", runtimeRec)
	}
	if _, ok := reloaded.Lookup("runtime-one"); !ok {
		t.Fatal("runtime record did not survive restart")
	}
	if _, ok := reloaded.Lookup("provisioned-one"); !ok {
		t.Fatal("provisioned record lost after restart")
	}
}

func TestExternalReplaceAlwaysFailsClosedWithoutSideEffects(t *testing.T) {
	root := t.TempDir()
	store := writeExternalRuntimeRecords(t, root, []ServableRecord{validRecord("legacy-one")})
	registry, err := NewExternal(ExternalConfig{ModelRoot: root, ProvisionFile: filepath.Join(t.TempDir(), "missing.yaml")})
	if err != nil {
		t.Fatal(err)
	}
	before := registry.Snapshot()
	beforeDisk, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}
	bad := validRecord("bad-one")
	bad.Endpoint = "no-port"
	candidates := [][]ServableRecord{
		nil,
		{},
		{validRecord("new-one")},
		{validRecord("legacy-one"), bad},
	}
	for _, candidate := range candidates {
		if err := registry.Replace(candidate); !errors.Is(err, ErrExternalServablesReadOnly) {
			t.Fatalf("Replace error = %v, want stable read-only sentinel", err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := registry.Replace([]ServableRecord{validRecord("concurrent-one")}); !errors.Is(err, ErrExternalServablesReadOnly) {
				t.Errorf("concurrent Replace error = %v", err)
			}
		}()
	}
	wg.Wait()
	if got := registry.Snapshot(); !reflect.DeepEqual(got, before) {
		t.Fatalf("read-only rejection changed merged view:\n%+v\n%+v", before, got)
	}
	afterDisk, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}
	if string(afterDisk) != string(beforeDisk) {
		t.Fatal("read-only rejection changed persisted legacy runtime bytes")
	}
	if _, ok := registry.Lookup("bad-one"); ok {
		t.Fatal("invalid candidate entered the merged view")
	}
	if _, ok := registry.Lookup("legacy-one"); !ok {
		t.Fatal("legacy runtime record was removed")
	}
}

func TestExternalProbeFailureStabilizesToFailed(t *testing.T) {
	root := t.TempDir()
	writeExternalRuntimeRecords(t, root, []ServableRecord{validRecord("down-one")})
	prober := &fakeExternalProber{err: errors.New("connection refused")}
	registry, err := NewExternal(ExternalConfig{ModelRoot: root, Prober: prober})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < externalStableCount; i++ {
		registry.scan(context.Background())
	}
	got := registry.Snapshot()
	if len(got) != 1 || got[0].State != "FAILED" || got[0].Reason != externalReasonRuntimeUnreach {
		t.Fatalf("want FAILED/%s, got %+v", externalReasonRuntimeUnreach, got)
	}
}
