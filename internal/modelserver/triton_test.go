// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

// fakeTritonControl implements the subset of the Triton HTTP Repository API
// and ModelReady endpoint the reconciler uses.
type fakeTritonControl struct {
	mu       sync.Mutex
	loaded   map[string]bool
	calls    []string
	failLoad map[string]bool
}

func newFakeTritonControl() *fakeTritonControl {
	return &fakeTritonControl{
		loaded:   map[string]bool{},
		failLoad: map[string]bool{},
	}
}

func (f *fakeTritonControl) server() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		switch {
		case strings.HasPrefix(r.URL.Path, "/v2/repository/models/") && strings.HasSuffix(r.URL.Path, "/load"):
			name := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v2/repository/models/"), "/load")
			f.calls = append(f.calls, "load:"+name)
			fail := f.failLoad[name]
			if fail {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			f.loaded[name] = true
			w.WriteHeader(http.StatusOK)
		case strings.HasPrefix(r.URL.Path, "/v2/repository/models/") && strings.HasSuffix(r.URL.Path, "/unload"):
			name := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v2/repository/models/"), "/unload")
			f.calls = append(f.calls, "unload:"+name)
			delete(f.loaded, name)
			w.WriteHeader(http.StatusOK)
		case strings.HasPrefix(r.URL.Path, "/v2/models/") && strings.HasSuffix(r.URL.Path, "/ready"):
			name := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/v2/models/"), "/ready")
			if f.loaded[name] {
				w.WriteHeader(http.StatusOK)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		default:
			http.NotFound(w, r)
		}
	}))
}

func (f *fakeTritonControl) hasCall(want string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, call := range f.calls {
		if call == want {
			return true
		}
	}
	return false
}

func (f *fakeTritonControl) isLoaded(name string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.loaded[name]
}

func (f *fakeTritonControl) callCount(prefix string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, call := range f.calls {
		if strings.HasPrefix(call, prefix) {
			n++
		}
	}
	return n
}

// startTritonTestCatalog installs one triton-family package. payloadPrefix
// selects the repository payload directory declared in the descriptor.
func startTritonTestCatalog(t *testing.T, manifest string, files map[string]string) (string, *modelcatalog.Catalog, string) {
	t.Helper()
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "triton.tar.gz")
	f, err := os.Create(pkg)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	writeOVMSTarFile(t, tw, "nereid-model.json", []byte(manifest))
	for name, content := range files {
		writeOVMSTarFile(t, tw, name, []byte(content))
	}
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
	t.Cleanup(catalog.Stop)
	return root, catalog, pkg
}

func tritonManifest(family, artifactFormat, repositoryPath, servedBy string, payloadFiles map[string]string) string {
	labels := []byte("labels\n")
	labelSum := sha256.Sum256(labels)
	var entries []string
	entries = append(entries, fmt.Sprintf(`{"path":"labels.txt","bytes":%d,"sha256":"%s"}`, len(labels), hex.EncodeToString(labelSum[:])))
	names := make([]string, 0, len(payloadFiles))
	for name := range payloadFiles {
		names = append(names, name)
	}
	sortStrings(names)
	for _, name := range names {
		if name == "labels.txt" {
			continue
		}
		content := []byte(payloadFiles[name])
		sum := sha256.Sum256(content)
		entries = append(entries, fmt.Sprintf(`{"path":%q,"bytes":%d,"sha256":"%s"}`, name, len(content), hex.EncodeToString(sum[:])))
	}
	served := ""
	if servedBy != "" {
		served = fmt.Sprintf(`,"served_by":[%s]`, servedBy)
	}
	repoLine := ""
	if repositoryPath != "" {
		repoLine = fmt.Sprintf(`,"repository_path":%q`, repositoryPath)
	}
	return fmt.Sprintf(`{
  "schema": "nereid-model/1",
  "model": {"id": "test-detect", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
  "artifact": {"format": %q, "precision": "FP16", "files": [%s]},
  "runtime": {
    "backend": "kserve",
    "model_name": "detect-model"%s,
    "server": {"family": %q}%s
  },
  "io": {
    "inputs": [{"name": "images", "shape": [1,3,10,10], "layout": "NCHW", "dtype": "FP32",
      "preprocess": {"resize": "stretch", "scale": "1/255", "color_order": "RGB"}}],
    "outputs": [{"name": "output", "shape": [1,6,1], "layout": "NCA", "dtype": "FP32"}]
  },
  "decoder": {"kind": "detection.anchors", "output": "output", "threshold": 0.25}
}`, artifactFormat, strings.Join(entries, ","), repoLine, family, served)
}

func sortStrings(in []string) {
	for i := 1; i < len(in); i++ {
		for j := i; j > 0 && in[j-1] > in[j]; j-- {
			in[j-1], in[j] = in[j], in[j-1]
		}
	}
}

func newTritonManager(t *testing.T, root string, catalog *modelcatalog.Catalog, control *httptest.Server) *TritonManager {
	t.Helper()
	m, err := NewTriton(TritonConfig{
		ModelRoot:       root,
		ContainerRoot:   "/models",
		ControlEndpoint: control.URL,
		GRPCEndpoint:    "edgex-triton:8001",
		Catalog:         catalog,
		LoadTimeout:     5 * time.Second,
		ReadyTimeout:    5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	m.Start()
	t.Cleanup(m.Stop)
	return m
}

func TestTritonAcquireRendersRepositoryAndLoads(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                           "labels\n",
		"repository/detect-model/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "onnx", "repository/detect-model", "", files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)

	entry, ok := catalog.Lookup("test-detect", "1.0.0")
	if !ok {
		t.Fatal("package was not installed")
	}
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer lease.Release()

	wantName := ServedName("test-detect", "1.0.0", "CPU")
	if lease.ModelName() != wantName {
		t.Fatalf("lease model = %q, want %q", lease.ModelName(), wantName)
	}
	if lease.Endpoint() != "edgex-triton:8001" {
		t.Fatalf("lease endpoint = %q", lease.Endpoint())
	}
	if !control.hasCall("load:" + wantName) {
		t.Fatalf("triton load was not called; calls=%v", control.calls)
	}

	dir := filepath.Join(root, filepath.FromSlash(tritonRepoRelPath), wantName)
	pbtxt, err := os.ReadFile(filepath.Join(dir, "config.pbtxt"))
	if err != nil {
		t.Fatal(err)
	}
	configText := string(pbtxt)
	for _, want := range []string{`name: "` + wantName + `"`, `backend: "onnxruntime"`, "max_batch_size: 0", "kind: KIND_CPU"} {
		if !strings.Contains(configText, want) {
			t.Fatalf("config.pbtxt missing %q:\n%s", want, configText)
		}
	}
	target, err := os.Readlink(filepath.Join(dir, "1"))
	if err != nil {
		t.Fatalf("version symlink: %v", err)
	}
	resolved, err := filepath.EvalSymlinks(filepath.Join(dir, "1"))
	if err != nil {
		t.Fatalf("resolve version symlink: %v", err)
	}
	if !strings.HasSuffix(resolved, filepath.Join("repository", "detect-model", "1")) {
		t.Fatalf("symlink %q resolves to %q", target, resolved)
	}

	// A second binding shares the servable: no additional load call.
	second, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	defer second.Release()
	if control.callCount("load:") != 1 {
		t.Fatalf("loads = %d, want 1 shared load", control.callCount("load:"))
	}
}

func TestTritonGPUInstanceGroupAndDeviceValidation(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                          "labels\n",
		"repository/detect-model/1/model.xml": "fake-model-bytes",
		"repository/detect-model/1/model.bin": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "openvino-ir", "repository/detect-model", "", files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)

	entry, _ := catalog.Lookup("test-detect", "1.0.0")
	lease, err := manager.Acquire(context.Background(), entry, "gpu.1")
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer lease.Release()

	dir := filepath.Join(root, filepath.FromSlash(tritonRepoRelPath), lease.ModelName())
	pbtxt, err := os.ReadFile(filepath.Join(dir, "config.pbtxt"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(pbtxt), "kind: KIND_GPU, gpus: [ 1 ]") {
		t.Fatalf("config.pbtxt missing GPU instance group:\n%s", pbtxt)
	}
	if !strings.Contains(string(pbtxt), `backend: "openvino"`) {
		t.Fatalf("config.pbtxt missing openvino backend:\n%s", pbtxt)
	}

	if _, err := manager.Acquire(context.Background(), entry, "MULTI:GPU,CPU"); err == nil {
		t.Fatal("compound MULTI target must be rejected for triton")
	}
}

func TestTritonReleaseUnloadsAndRemovesRepository(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                           "labels\n",
		"repository/detect-model/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "onnx", "repository/detect-model", "", files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)

	entry, _ := catalog.Lookup("test-detect", "1.0.0")
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	name := lease.ModelName()
	lease.Release()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if control.hasCall("unload:" + name) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !control.hasCall("unload:" + name) {
		t.Fatalf("unload was not called; calls=%v", control.calls)
	}
	dir := filepath.Join(root, filepath.FromSlash(tritonRepoRelPath), name)
	deadline = time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("rendered directory %s still exists after release", dir)
}

func TestTritonServedByPayloadAndBackendHint(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	slot := `{"family":"triton","backend_hint":"onnxruntime","payload":"payloads/triton-onnx"}`
	files := map[string]string{
		"labels.txt":                        "labels\n",
		"payloads/triton-onnx/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "onnx", "", slot, files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)

	entry, _ := catalog.Lookup("test-detect", "1.0.0")
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer lease.Release()

	dir := filepath.Join(root, filepath.FromSlash(tritonRepoRelPath), lease.ModelName())
	resolved, err := filepath.EvalSymlinks(filepath.Join(dir, "1"))
	if err != nil {
		t.Fatalf("resolve served_by symlink: %v", err)
	}
	if !strings.HasSuffix(resolved, filepath.Join("payloads", "triton-onnx", "1")) {
		t.Fatalf("served_by payload not linked, got %q", resolved)
	}
	pbtxt, err := os.ReadFile(filepath.Join(dir, "config.pbtxt"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(pbtxt), `backend: "onnxruntime"`) {
		t.Fatalf("backend hint not rendered:\n%s", pbtxt)
	}
}

func TestTritonLoadFailureReportsErrorAndRetries(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                           "labels\n",
		"repository/detect-model/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "onnx", "repository/detect-model", "", files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)
	entry, _ := catalog.Lookup("test-detect", "1.0.0")

	name := ServedName("test-detect", "1.0.0", "CPU")
	control.failLoad[name] = true
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatalf("load failure must keep the lease and rely on readiness gating: %v", err)
	}
	defer lease.Release()
	control.mu.Lock()
	isLoaded := control.loaded[name]
	control.mu.Unlock()
	if isLoaded {
		t.Fatal("servable must not be marked loaded while the server rejects load")
	}
	if control.callCount("unload:") != 0 {
		t.Fatal("failed new load must not unload any previous generation")
	}

	// Recovery: clear the failure and let the reconciler retry loop recover.
	control.mu.Lock()
	delete(control.failLoad, name)
	control.mu.Unlock()

	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		if control.hasCall("load:"+name) && control.isLoaded(name) {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	control.mu.Lock()
	ready := control.loaded[name]
	control.mu.Unlock()
	if !ready {
		t.Fatalf("servable did not recover after load failure; calls=%v", control.calls)
	}
}

func TestTritonRemoveModelUnloadsBeforeDeletion(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                           "labels\n",
		"repository/detect-model/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("triton", "onnx", "repository/detect-model", "", files)
	root, catalog, pkg := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)
	entry, _ := catalog.Lookup("test-detect", "1.0.0")
	lease, err := manager.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	name := lease.ModelName()

	manager.RemoveModel("test-detect", "1.0.0")
	if lease.Current() {
		t.Fatal("lease generation must be invalidated by RemoveModel")
	}
	if !control.hasCall("unload:" + name) {
		t.Fatalf("RemoveModel did not unload; calls=%v", control.calls)
	}
	lease.Release() // idempotent no-op against the stale generation
	if err := os.Remove(pkg); err != nil {
		t.Fatal(err)
	}
}

func TestTritonTargetDeviceMatrixDoesNotFallBackToCPU(t *testing.T) {
	for input, want := range map[string]string{
		"":        "CPU",
		" cpu ":   "CPU",
		"gpu":     "GPU",
		" gpu.0 ": "GPU.0",
		"GPU.12":  "GPU.12",
	} {
		got, err := normalizeTritonDevice(input)
		if err != nil || got != want {
			t.Errorf("normalizeTritonDevice(%q) = %q, %v; want %q", input, got, err, want)
		}
	}
	for _, input := range []string{"GPU.-1", "GPU.zero", "MULTI:CPU,GPU", "NPU", "CPU;GPU"} {
		got, err := normalizeTritonDevice(input)
		if err == nil {
			t.Errorf("normalizeTritonDevice(%q) = %q; unsupported targets must not fall back to CPU", input, got)
		}
	}
	if index, gpu := tritonGPUSelector("GPU.3"); !gpu || index != 3 {
		t.Fatalf("GPU.3 selector = index %d, gpu %v; want 3/true", index, gpu)
	}
	if _, gpu := tritonGPUSelector("CPU"); gpu {
		t.Fatal("CPU must select KIND_CPU, never an implicit GPU")
	}
}

func TestTritonRejectsNonTritonPackages(t *testing.T) {
	control := newFakeTritonControl()
	server := control.server()
	defer server.Close()

	files := map[string]string{
		"labels.txt":                           "labels\n",
		"repository/detect-model/1/model.onnx": "fake-model-bytes",
	}
	manifest := tritonManifest("ovms", "onnx", "repository/detect-model", "", files)
	root, catalog, _ := startTritonTestCatalog(t, manifest, files)
	manager := newTritonManager(t, root, catalog, server)
	entry, _ := catalog.Lookup("test-detect", "1.0.0")
	if _, err := manager.Acquire(context.Background(), entry, "CPU"); err == nil {
		t.Fatal("OVMS package must be rejected by the triton adapter")
	}
}
