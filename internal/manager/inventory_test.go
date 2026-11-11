// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package manager

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"

	"github.com/edgexfoundry/device-inference/internal/infer"
	pb "github.com/edgexfoundry/device-inference/internal/infer/kserve/grpcv2"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

const testInventoryManifest = `{
  "schema": "nereid-model/1",
  "model": {"id": "test-model", "version": "1.0.0", "revision": "76937", "task": "detection", "family": "yolov8", "labels_file": "labels.txt"},
  "artifact": {"format": "openvino-ir", "precision": "FP32", "files": [
    {"path": "labels.txt", "bytes": %[1]d, "sha256": "%[2]s"}
  ]},
  "runtime": {"backend": "kserve", "model_name": "test-model-runtime",
    "server": {"family": "triton"}},
  "io": {
    "inputs": [{"name": "images", "shape": [1,3,2,2], "layout": "NCHW", "dtype": "FP32",
      "preprocess": {"resize": "stretch", "scale": "1/255"}}],
    "outputs": [{"name": "output", "shape": [1,2], "layout": "NC", "dtype": "FP32"}]
  },
  "decoder": {"kind": "classification.probs", "output": "output"}
}`

type testReadyServer struct {
	pb.UnimplementedGRPCInferenceServiceServer
	ready bool
}

func (s *testReadyServer) ModelReady(_ context.Context, _ *pb.ModelReadyRequest) (*pb.ModelReadyResponse, error) {
	return &pb.ModelReadyResponse{Ready: s.ready}, nil
}

func TestInventoryWaitsForStableRuntimeResult(t *testing.T) {
	catalog, cleanup := startTestCatalog(t)
	defer cleanup()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(server, &testReadyServer{ready: true})
	go func() { _ = server.Serve(listener) }()
	defer server.Stop()

	backends := infer.NewRegistry("", nil)
	defer backends.Close()
	inv := New(Config{
		Catalog:        catalog,
		Backends:       backends,
		KServeEndpoint: listener.Addr().String(),
	})
	defer inv.Stop()

	models := inv.Models(nil)
	if len(models) != 1 || models[0].State != model.StateValidating {
		t.Fatalf("initial inventory = %+v, want one VALIDATING model", models)
	}

	for i := 0; i < stableProbeCount; i++ {
		inv.scan(t.Context())
	}
	models = inv.Models(map[string]int{BindingKey("test-model", "1.0.0"): 2})
	if len(models) != 1 {
		t.Fatalf("models = %+v", models)
	}
	got := models[0]
	if got.State != model.StateReady || got.Reason != "" || got.ActiveBindings != 2 {
		t.Fatalf("model after stable probes = %+v, want READY with 2 bindings", got)
	}
	if got.ID != "test-model" || got.Version != "1.0.0" || got.Revision != "76937" ||
		got.Task != "detection" || got.Family != "yolov8" || got.Backend != "kserve" ||
		got.ModelName != "test-model-runtime" || len(got.Labels) != 2 {
		t.Fatalf("unexpected inventory fields: %+v", got)
	}
}

func TestInventoryReportsModelNotReady(t *testing.T) {
	catalog, cleanup := startTestCatalog(t)
	defer cleanup()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(server, &testReadyServer{ready: false})
	go func() { _ = server.Serve(listener) }()
	defer server.Stop()

	backends := infer.NewRegistry("", nil)
	defer backends.Close()
	inv := New(Config{
		Catalog:        catalog,
		Backends:       backends,
		KServeEndpoint: listener.Addr().String(),
	})
	defer inv.Stop()

	for i := 0; i < stableProbeCount; i++ {
		inv.scan(t.Context())
	}
	models := inv.Models(nil)
	if len(models) != 1 {
		t.Fatalf("models = %+v", models)
	}
	if models[0].State != model.StateFailed || models[0].Reason != "model-not-ready" {
		t.Fatalf("model = %+v, want FAILED/model-not-ready", models[0])
	}
}

func startTestCatalog(t *testing.T) (*modelcatalog.Catalog, func()) {
	t.Helper()
	root := t.TempDir()
	inbox := t.TempDir()
	pkg := filepath.Join(inbox, "model.tar.gz")
	writeTestPackage(t, pkg)
	catalog, err := modelcatalog.New(modelcatalog.Config{
		Root:         root,
		InboxEnabled: true,
		InboxDir:     inbox,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	return catalog, func() { catalog.Stop() }
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
	sum := sha256.Sum256(labels)
	writeTarFile(t, tw, "nereid-model.json",
		[]byte(fmt.Sprintf(testInventoryManifest, len(labels), hex.EncodeToString(sum[:]))))
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
