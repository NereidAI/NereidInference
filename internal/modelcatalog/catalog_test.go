// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelcatalog

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func sha256Text(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

const testManifest = `{
  "schema": "nereid-model/1",
  "model": {"id": "test-model", "version": "1.0.0", "task": "classification", "labels_file": "labels.txt"},
  "artifact": {"format": "openvino-ir", "precision": "FP32", "files": [
    {"path": "labels.txt", "bytes": %[1]d, "sha256": "%[2]s"}
  ]},
  "runtime": {"backend": "kserve", "model_name": "test-model",
    "server": {"family": "ovms"}},
  "io": {
    "inputs": [{"name": "images", "shape": [1,3,2,2], "layout": "NCHW", "dtype": "FP32",
      "preprocess": {"resize": "stretch", "scale": "1/255"}}],
    "outputs": [{"name": "output", "shape": [1,2], "layout": "NC", "dtype": "FP32"}]
  },
  "decoder": {"kind": "classification.probs", "output": "output"}
}`

func TestIngestAndRemovePackage(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "model.tar.gz")
	writeTestPackage(t, pkg)

	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()

	entry, ok := c.Lookup("test-model", "1.0.0")
	if !ok {
		t.Fatal("package was not installed")
	}
	if entry.Dir == "" || len(entry.Labels) != 2 || entry.Labels[0] != "a" {
		t.Fatalf("unexpected entry: %+v labels=%v", entry, entry.Labels)
	}
	if _, err := os.Stat(filepath.Join(entry.Dir, "nereid-model.json")); err != nil {
		t.Fatal(err)
	}

	hookSawModel := false
	c.cfg.BeforeRemove = func(id, version string) {
		if id == "test-model" && version == "1.0.0" {
			if _, err := os.Stat(filepath.Join(entry.Dir, "nereid-model.json")); err == nil {
				hookSawModel = true
			}
		}
	}

	if err := os.Remove(pkg); err != nil {
		t.Fatal(err)
	}
	c.handleRemoval(pkg)
	if !hookSawModel {
		t.Fatal("before-remove hook did not observe the active model before files were deleted")
	}
	if _, ok := c.Lookup("test-model", "1.0.0"); ok {
		t.Fatal("removed package still resolves")
	}
	if _, err := os.Stat(filepath.Join(root, "active", "test-model")); !os.IsNotExist(err) {
		t.Fatalf("empty active model parent should be removed, stat err=%v", err)
	}
}

func TestStopBeforeStartIsSafe(t *testing.T) {
	c, err := New(Config{Root: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	c.Stop()
	c.Stop()
}

func TestRemovedEventUnblocksWhenSubscriberReleases(t *testing.T) {
	c := &Catalog{
		stop: make(chan struct{}),
		subs: map[*eventSubscription]struct{}{},
	}
	sub := &eventSubscription{ch: make(chan Event, 16), done: make(chan struct{})}
	c.subs[sub] = struct{}{}
	release := func() {
		c.subMu.Lock()
		delete(c.subs, sub)
		close(sub.done)
		c.subMu.Unlock()
	}
	for i := 0; i < cap(sub.ch); i++ {
		sub.ch <- Event{Kind: EventRemoved}
	}
	done := make(chan struct{})
	go func() {
		c.publish(Event{Kind: EventRemoved, ModelID: "m", Version: "1"})
		close(done)
	}()
	release()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("removal publish did not observe unsubscribe")
	}
}

func TestPrunesSourceDeletedWhileStopped(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "model.tar.gz")
	writeTestPackage(t, pkg)

	c1, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c1.Start(); err != nil {
		t.Fatal(err)
	}
	entry, ok := c1.Lookup("test-model", "1.0.0")
	if !ok {
		t.Fatal("package was not installed")
	}
	c1.Stop()
	if err := os.Remove(pkg); err != nil {
		t.Fatal(err)
	}

	c2, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c2.Start(); err != nil {
		t.Fatal(err)
	}
	defer c2.Stop()
	if _, ok := c2.Lookup("test-model", "1.0.0"); ok {
		t.Fatal("source deleted while stopped was not pruned")
	}
	if _, err := os.Stat(entry.Dir); !os.IsNotExist(err) {
		t.Fatalf("active directory still exists, stat err=%v", err)
	}
}

func TestRestartDoesNotReextractUnchangedPackage(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "model.tar.gz")
	writeTestPackage(t, pkg)

	c1, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c1.Start(); err != nil {
		t.Fatal(err)
	}
	entry, ok := c1.Lookup("test-model", "1.0.0")
	if !ok {
		t.Fatal("package was not installed")
	}
	sentinel := filepath.Join(entry.Dir, "restart-sentinel")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	c1.Stop()

	c2, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c2.Start(); err != nil {
		t.Fatal(err)
	}
	defer c2.Stop()
	if _, err := os.Stat(sentinel); err != nil {
		t.Fatalf("unchanged package was re-installed and removed sentinel: %v", err)
	}
}

func TestProbesNonTarObjectOnce(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	object := filepath.Join(inbox, "not-a-model")
	if err := os.WriteFile(object, []byte("plain object"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()

	if err := c.ingestFile(object); err != nil {
		t.Fatalf("non-model object should be ignored, got %v", err)
	}
	if _, ok := c.Lookup("test-model", ""); ok {
		t.Fatal("non-model object was installed")
	}
}

func TestRejectsBareONNXTarBecausePackageMustBeGzip(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	object := filepath.Join(inbox, "bare-onnx")
	f, err := os.Create(object)
	if err != nil {
		t.Fatal(err)
	}
	tw := tar.NewWriter(f)
	writeTarFile(t, tw, "model.onnx", []byte("not really onnx"))
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()
	if _, ok := c.Lookup("model", ""); ok {
		t.Fatal("bare ONNX tar must not bypass the standard gzip package contract")
	}
}

func TestRejectsUndeclaredPayloadFile(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	manifest := []byte(`{
	  "schema": "nereid-model/1",
	  "model": {"id": "m", "version": "1.0.0", "task": "classification", "labels_file": "labels.txt"},
	  "artifact": {"format": "openvino-ir", "precision": "FP16", "files": []},
	  "runtime": {"backend": "kserve", "model_name": "m", "server": {"family": "ovms"}},
	  "io": {
	    "inputs": [{"name": "images", "shape": [1,3,2,2], "layout": "NCHW", "dtype": "FP32"}],
	    "outputs": [{"name": "o", "shape": [1,2], "layout": "NC", "dtype": "FP32"}]
	  },
	  "decoder": {"kind": "classification.probs", "output": "o"}
	}`)
	pkg := filepath.Join(inbox, "m.tar.gz")
	writeGenericPackage(t, pkg, manifest, map[string][]byte{
		"labels.txt": []byte("a\nb\n"),
	})
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()
	if _, ok := c.Lookup("m", "1.0.0"); ok {
		t.Fatal("package with an undeclared payload file must be rejected")
	}
	var sawFailed bool
	for _, o := range c.ObjectDiagnostics().Objects {
		if o.State == "FAILED" {
			sawFailed = true
		}
	}
	if !sawFailed {
		t.Fatal("undeclared payload should be recorded as FAILED in object diagnostics")
	}
}

func writeGenericPackage(t *testing.T, p string, manifest []byte, files map[string][]byte) {
	t.Helper()
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	writeTarFile(t, tw, "nereid-model.json", manifest)
	for _, name := range names {
		writeTarFile(t, tw, name, files[name])
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, time.Now(), time.Now()); err != nil {
		t.Fatal(err)
	}
}

func floatManifest(id, version, revision string) []byte {
	rev := ""
	if revision != "" {
		rev = fmt.Sprintf(`,"revision":%q`, revision)
	}
	return []byte(fmt.Sprintf(`{
	  "schema": "nereid-model/1",
	  "model": {"id": %q, "version": %q%s, "task": "regression"},
	  "artifact": {"format": "openvino-ir", "precision": "FP16", "files": []},
	  "runtime": {"backend": "kserve", "model_name": "m", "server": {"family": "ovms"}},
	  "io": {
	    "inputs": [{"name": "images", "shape": [1,3,2,2], "layout": "NCHW", "dtype": "FP32"}],
	    "outputs": [{"name": "values", "shape": [1,2], "layout": "NC", "dtype": "FP32"}]
	  },
	  "decoder": {"kind": "regression.values", "output": "values",
	    "names": [{"channel": 0, "name": "x"}]}
	}`, id, version, rev))
}

func waitLookup(t *testing.T, c *Catalog, id, version string, want bool) *Entry {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if e, ok := c.Lookup(id, version); ok == want {
			return e
		}
		if time.Now().After(deadline) {
			t.Fatalf("Lookup(%q,%q) want ok=%v not reached", id, version, want)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestFloatingResolutionPrefersHigherPrecision(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	writeGenericPackage(t, filepath.Join(inbox, "int8.tar.gz"),
		floatManifest("m", "1.0.0-int8", "102"), nil)
	writeGenericPackage(t, filepath.Join(inbox, "fp16.tar.gz"),
		floatManifest("m", "1.0.0-fp16", "103"), nil)
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()
	waitLookup(t, c, "m", "1.0.0-fp16", true)
	waitLookup(t, c, "m", "1.0.0-int8", true)
	got := waitLookup(t, c, "m", "", true)
	if got.Version != "1.0.0-fp16" {
		t.Fatalf("floating resolution picked %q, want 1.0.0-fp16", got.Version)
	}
}

func TestFloatingResolutionSkipsCompiledPackages(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	writeGenericPackage(t, filepath.Join(inbox, "rk.tar.gz"),
		floatManifest("m", "1.0.0-int8-rk3588", "1"), nil)
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()
	waitLookup(t, c, "m", "1.0.0-int8-rk3588", true)
	if e, ok := c.Lookup("m", ""); ok {
		t.Fatalf("floating lookup must not select compiled package %q", e.Version)
	}
}

func TestCompareRevisionNumeric(t *testing.T) {
	if got := compareRevision("10", "9"); got <= 0 {
		t.Fatalf("numeric revision compare = %d, want positive", got)
	}
	if got := compareRevision("9", "10"); got >= 0 {
		t.Fatalf("numeric revision compare = %d, want negative", got)
	}
	if got := compareRevision("abc", "abd"); got >= 0 {
		t.Fatalf("lexical revision compare = %d, want negative", got)
	}
}

func TestObjectDiagnosticsReportsIgnoredObject(t *testing.T) {
	root := t.TempDir()
	inbox := t.TempDir()
	object := filepath.Join(inbox, "business-object")
	if err := os.WriteFile(object, []byte("not an archive"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()
	diag := c.ObjectDiagnostics()
	if diag.IgnoredCount != 1 || len(diag.Objects) != 1 || diag.Objects[0].State != "IGNORED" {
		t.Fatalf("unexpected diagnostics: %+v", diag)
	}
}

func writeTestPackage(t *testing.T, path string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	labels := []byte("a\nb\n")
	writeTarFile(t, tw, "nereid-model.json", []byte(fmt.Sprintf(testManifest, len(labels), sha256Text(labels))))
	writeTarFile(t, tw, "labels.txt", labels)
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, time.Now(), time.Now()); err != nil {
		t.Fatal(err)
	}
}

func writeTarFile(t *testing.T, tw *tar.Writer, name string, b []byte) {
	t.Helper()
	if err := tw.WriteHeader(&tar.Header{
		Name: name, Mode: 0o644, Size: int64(len(b)), Typeflag: tar.TypeReg,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(b); err != nil {
		t.Fatal(err)
	}
}
