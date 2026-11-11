// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

func testOVMSManifest(xml, bin, labels []byte) string {
	xmlSum := sha256.Sum256(xml)
	binSum := sha256.Sum256(bin)
	labelsSum := sha256.Sum256(labels)
	return fmt.Sprintf(`{
  "schema": "nereid-model/1",
  "model": {"id": "test-detect", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
  "artifact": {"format": "openvino-ir", "precision": "FP16", "files": [
    {"path":"labels.txt","bytes":%[4]d,"sha256":"%[5]s"},
    {"path":"repository/detect-model/1/model.xml","bytes":%[1]d,"sha256":"%[2]s"},
    {"path":"repository/detect-model/1/model.bin","bytes":%[3]d,"sha256":"%[6]s"}
  ]},
  "runtime": {
    "backend": "kserve",
    "model_name": "detect-model",
    "repository_path": "repository/detect-model",
    "server": {"family": "ovms", "min_version": "2026.2.1"}
  },
  "io": {
    "inputs": [{"name": "images", "shape": [1,3,10,10], "layout": "NCHW", "dtype": "FP32"}],
    "outputs": [{"name": "output", "shape": [1,6,1], "layout": "NCA", "dtype": "FP32"}]
  },
  "decoder": {"kind": "detection.anchors", "output": "output"}
}`,
		len(xml), hex.EncodeToString(xmlSum[:]), len(bin),
		len(labels), hex.EncodeToString(labelsSum[:]), hex.EncodeToString(binSum[:]))
}

func TestOVMSArtifactCompatibility(t *testing.T) {
	t.Run("onnx", func(t *testing.T) {
		entry := newOVMSLayoutTestEntry(t, model.ArtifactFormatONNX, map[string]string{
			"repository/m/1/model.onnx": "weights",
		})
		if err := validateOVMSArtifact(entry); err != nil {
			t.Fatalf("ONNX layout rejected: %v", err)
		}
	})
	t.Run("openvino ir requires xml and bin", func(t *testing.T) {
		entry := newOVMSLayoutTestEntry(t, model.ArtifactFormatOpenVINOIR, map[string]string{
			"repository/m/1/model.xml": "<net/>",
		})
		if err := validateOVMSArtifact(entry); err == nil {
			t.Fatal("OpenVINO IR layout without model.bin was accepted")
		}
	})
	t.Run("future format is adapter rejected not catalog rejected", func(t *testing.T) {
		entry := newOVMSLayoutTestEntry(t, "tensorrt-engine", map[string]string{
			"repository/m/1/model.plan": "engine",
		})
		if err := validateOVMSArtifact(entry); err == nil {
			t.Fatal("OVMS accepted a TensorRT engine artifact")
		}
	})
	t.Run("only one version directory named 1", func(t *testing.T) {
		entry := newOVMSLayoutTestEntry(t, model.ArtifactFormatOpenVINOIR, map[string]string{
			"repository/m/1/model.xml": "<net/>",
			"repository/m/1/model.bin": "weights",
			"repository/m/2/model.xml": "<net/>",
			"repository/m/2/model.bin": "weights2",
		})
		if err := validateOVMSArtifact(entry); err == nil {
			t.Fatal("OVMS accepted a repository with an extra version directory")
		}
	})
}

func newOVMSLayoutTestEntry(t *testing.T, format string, files map[string]string) *modelcatalog.Entry {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	manifest := &model.Manifest{
		Artifact: model.ArtifactSection{Format: format},
		Runtime: model.RuntimeSection{
			Backend:        "kserve",
			ModelName:      "m",
			RepositoryPath: "repository/m",
		},
	}
	manifest.Runtime.Server.Family = ovmsFamily
	return &modelcatalog.Entry{ID: "m", Version: "1.0.0", Dir: dir, Manifest: manifest}
}

func TestOVMSManagerRequiresGRPCEndpoint(t *testing.T) {
	root, catalog, _, cleanup := startOVMSTestCatalog(t)
	defer cleanup()
	if _, err := New(Config{ModelRoot: root, Catalog: catalog}); err == nil {
		t.Fatal("managed OVMS must not start without its lease gRPC endpoint")
	}
}

func TestOVMSManagerAcquiresAndReleasesBindingServable(t *testing.T) {
	root, catalog, pkg, cleanup := startOVMSTestCatalog(t)
	defer cleanup()

	reloads := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/config/reload" {
			t.Errorf("unexpected reload request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		reloads++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	manager, err := New(Config{
		ModelRoot:      root,
		ContainerRoot:  "/models",
		ReloadEndpoint: server.URL,
		GRPCEndpoint:   "ovms:9000",
		Catalog:        catalog,
		ReloadTimeout:  time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()

	entry, ok := catalog.Lookup("test-detect", "1.0.0")
	if !ok {
		t.Fatal("model was not installed")
	}
	lease, err := manager.Acquire(context.Background(), entry, "gpu.0")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	if got := lease.Endpoint(); got != "ovms:9000" {
		t.Fatalf("lease endpoint = %q, want manager-owned OVMS endpoint", got)
	}

	configPath := filepath.Join(root, "runtime", "ovms", "config.json")
	got := readOVMSConfig(t, configPath)
	if len(got.ModelConfigList) != 1 {
		t.Fatalf("config entries = %d, want 1: %+v", len(got.ModelConfigList), got)
	}
	cfg := got.ModelConfigList[0].Config
	wantName := ServedName("test-detect", "1.0.0", "GPU.0")
	if cfg["name"] != wantName {
		t.Fatalf("name = %v, want %q", cfg["name"], wantName)
	}
	if cfg["target_device"] != "GPU.0" {
		t.Fatalf("target_device = %v", cfg["target_device"])
	}
	if cfg["base_path"] != "/models/active/test-detect/1.0.0/repository/detect-model" {
		t.Fatalf("base_path = %v", cfg["base_path"])
	}
	if reloads == 0 {
		t.Fatal("OVMS reload endpoint was not called after acquire")
	}

	lease.Release()
	got = readOVMSConfig(t, configPath)
	if len(got.ModelConfigList) != 0 {
		t.Fatalf("config after release = %+v, want empty list", got)
	}
	if reloads < 2 {
		t.Fatalf("reloads = %d, want reload after release", reloads)
	}

	// Keep the deferred release idempotent.
	_ = pkg
}

func TestOVMSManagerPrunesServableWhenDistributionIsDeleted(t *testing.T) {
	root, catalog, pkg, cleanup := startOVMSTestCatalog(t)
	defer cleanup()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	manager, err := New(Config{
		ModelRoot:      root,
		ContainerRoot:  "/models",
		ReloadEndpoint: server.URL,
		GRPCEndpoint:   "ovms:9000",
		Catalog:        catalog,
		ReloadTimeout:  time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()

	entry, ok := catalog.Lookup("test-detect", "1.0.0")
	if !ok {
		t.Fatal("model was not installed")
	}
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	if err := os.Remove(pkg); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(root, "runtime", "ovms", "config.json")
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if len(readOVMSConfig(t, configPath).ModelConfigList) == 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("deleted distribution still configured: %+v", readOVMSConfig(t, configPath))
}

func TestLeaseGoesStaleWhenModelRedistributed(t *testing.T) {
	root, catalog, _, cleanup := startOVMSTestCatalog(t)
	defer cleanup()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	manager, err := New(Config{
		ModelRoot:      root,
		ContainerRoot:  "/models",
		ReloadEndpoint: server.URL,
		GRPCEndpoint:   "ovms:9000",
		Catalog:        catalog,
		ReloadTimeout:  time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()

	entry, ok := catalog.Lookup("test-detect", "1.0.0")
	if !ok {
		t.Fatal("model was not installed")
	}
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	if !lease.Current() {
		t.Fatal("freshly acquired lease is not current")
	}

	// Pre-removal hook fires when the package is withdrawn (catalog churn,
	// redistribution) even while the job goroutine keeps its lease pointer.
	manager.RemoveModel("test-detect", "1.0.0")
	if lease.Current() {
		t.Fatal("lease stayed current after model removal")
	}
	// The stale Release must not decrement the new generation.
	lease.Release()

	// When the package is available again a job re-Acquires and the servable
	// is rendered once more.
	fresh, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Release()
	if !fresh.Current() {
		t.Fatal("re-acquired lease is not current")
	}
	configPath := filepath.Join(root, "runtime", "ovms", "config.json")
	if got := readOVMSConfig(t, configPath); len(got.ModelConfigList) != 1 {
		t.Fatalf("config after re-acquire = %+v, want one servable", got)
	}
}

type testOVMSConfigFile struct {
	ModelConfigList []struct {
		Config map[string]any `json:"config"`
	} `json:"model_config_list"`
}

func readOVMSConfig(t *testing.T, path string) testOVMSConfigFile {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got testOVMSConfigFile
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("parse config %s: %v", string(b), err)
	}
	return got
}

func startOVMSTestCatalog(t *testing.T) (string, *modelcatalog.Catalog, string, func()) {
	t.Helper()
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "model.tar.gz")
	f, err := os.Create(pkg)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	xml := []byte("<net></net>")
	bin := []byte("weights")
	labels := []byte("background\nobject\n")
	writeOVMSTarFile(t, tw, "nereid-model.json", []byte(testOVMSManifest(xml, bin, labels)))
	writeOVMSTarFile(t, tw, "labels.txt", labels)
	writeOVMSTarFile(t, tw, "repository/detect-model/1/model.xml", xml)
	writeOVMSTarFile(t, tw, "repository/detect-model/1/model.bin", bin)
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	return root, catalog, pkg, func() {
		catalog.Stop()
	}
}

func writeOVMSTarFile(t *testing.T, tw *tar.Writer, name string, b []byte) {
	t.Helper()
	if err := tw.WriteHeader(&tar.Header{Name: name, Size: int64(len(b)), Mode: 0o644, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(b); err != nil {
		t.Fatal(err)
	}
}
