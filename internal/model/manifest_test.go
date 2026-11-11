// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadRejectsUnsafeManifest(t *testing.T) {
	for name, manifest := range map[string]string{
		"id traversal":      `{"schema":"nereid-model/1","model":{"id":"../evil","version":"1.0.0","task":"detection"},"runtime":{"backend":"kserve","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,2,2]}]}}`,
		"version separator": `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0/0","task":"detection"},"runtime":{"backend":"kserve","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,2,2]}]}}`,
		"revision space":    `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","revision":"bad name","task":"detection"},"runtime":{"backend":"kserve","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,2,2]}]}}`,
		"unknown backend":   `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","task":"detection"},"runtime":{"backend":"magic","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,2,2]}]}}`,
		"bad vision shape":  `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","task":"detection"},"runtime":{"backend":"kserve","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,4,2,2]}]}}`,
		"huge vision shape": `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","task":"detection"},"runtime":{"backend":"kserve","model_name":"m"},"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,4097,4097]}]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := Load(dir); err == nil {
				t.Fatal("expected invalid manifest to be rejected")
			}
		})
	}
}

func TestLoadRejectsContractV03Violations(t *testing.T) {
	base := `{
	  "schema": "nereid-model/1",
	  "model": {"id": "m", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
	  "artifact": {"format": "openvino-ir", %[1]s},
	  "runtime": {"backend": "kserve", "model_name": "m"%[2]s},
	  "io": {
	    "inputs": [{"name": "images", "shape": [1,3,640,640], "layout": "NCHW", "dtype": "FP32"}],
	    "outputs": [%[3]s]
	  },
	  "decoder": %[4]s
	}`
	for name, tc := range map[string]struct {
		artifact string
		runtime  string
		output   string
		decoder  string
	}{
		"missing server family": {
			artifact: `"precision": "FP16"`,
			output:   `{"name":"output0","shape":[1,6,8400],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"output0"}`,
		},
		"lowercase precision": {
			artifact: `"precision": "fp16"`,
			runtime:  `,"server":{"family":"ovms"}`,
			output:   `{"name":"output0","shape":[1,6,8400],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"output0"}`,
		},
		"dynamic anchors output": {
			artifact: `"precision": "FP16"`,
			runtime:  `,"server":{"family":"ovms"}`,
			output:   `{"name":"output0","shape":[1,6,-1],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"output0"}`,
		},
		"anchors output too few channels": {
			artifact: `"precision": "FP16"`,
			runtime:  `,"server":{"family":"ovms"}`,
			output:   `{"name":"output0","shape":[1,4,10],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"output0"}`,
		},
		"decoder references unknown output": {
			artifact: `"precision": "FP16"`,
			runtime:  `,"server":{"family":"ovms"}`,
			output:   `{"name":"output0","shape":[1,6,8400],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"missing"}`,
		},
		"zero output dimension": {
			artifact: `"precision": "FP16"`,
			runtime:  `,"server":{"family":"ovms"}`,
			output:   `{"name":"output0","shape":[1,6,0],"layout":"NCA","dtype":"FP32"}`,
			decoder:  `{"kind":"detection.anchors","output":"output0"}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			manifest := fmt.Sprintf(base, tc.artifact, tc.runtime, tc.output, tc.decoder)
			if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "labels.txt"), []byte("a\nb\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := Load(dir); err == nil {
				t.Fatal("expected manifest to be rejected")
			}
		})
	}
}

func TestAnchorsKindRequiresLabelsFile(t *testing.T) {
	dir := t.TempDir()
	manifest := `{
	  "schema": "nereid-model/1",
	  "model": {"id": "m", "version": "1.0.0", "task": "detection"},
	  "artifact": {"format": "openvino-ir", "precision": "FP16"},
	  "runtime": {"backend": "kserve", "model_name": "m", "server": {"family": "ovms"}},
	  "io": {
	    "inputs": [{"name": "images", "shape": [1,3,640,640], "layout": "NCHW", "dtype": "FP32"}],
	    "outputs": [{"name":"output0","shape":[1,6,8400],"layout":"NCA","dtype":"FP32"}]
	  },
	  "decoder": {"kind":"detection.anchors","output":"output0"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("anchors package without labels_file must be rejected")
	}
}

func TestLoadAcceptsStaticAnchorsPackage(t *testing.T) {
	dir := t.TempDir()
	manifest := `{
	  "schema": "nereid-model/1",
	  "model": {"id": "m", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
	  "artifact": {"format": "openvino-ir", "precision": "FP16"},
	  "runtime": {"backend": "kserve", "model_name": "m",
	    "repository_path": "repository/m", "server": {"family": "ovms"}},
	  "io": {
	    "inputs": [{"name": "images", "shape": [1,3,640,640], "layout": "NCHW", "dtype": "FP32"}],
	    "outputs": [{"name":"output0","role":"predictions","shape":[1,6,8400],"layout":"NCA","dtype":"FP32"}]
	  },
	  "decoder": {"kind":"detection.anchors","output":"predictions"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "labels.txt"), []byte("a\nb\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	m, _, err := Load(dir)
	if err != nil {
		t.Fatalf("static anchors package rejected: %v", err)
	}
	if m.Artifact.Precision != PrecisionFP16 {
		t.Fatalf("precision = %q", m.Artifact.Precision)
	}
}

func TestLoadAcceptsFutureRuntimeArtifactFormat(t *testing.T) {
	dir := t.TempDir()
	manifest := `{"schema":"nereid-model/1","model":{"id":"engine","version":"1.0.0","task":"detection"},
"artifact":{"format":"tensorrt-engine"},
"runtime":{"backend":"kserve","model_name":"engine","server":{"family":"triton"}},
"io":{"inputs":[{"name":"images","dtype":"FP32","layout":"NCHW","shape":[1,3,2,2]}]}}`
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	m, _, err := Load(dir)
	if err != nil {
		t.Fatalf("catalog-level manifest must not enforce OVMS formats: %v", err)
	}
	if m.Artifact.Format != "tensorrt-engine" || m.Runtime.Server.Family != "triton" {
		t.Fatalf("unexpected manifest: %+v", m)
	}
}

func TestLoadAcceptsOpenAIManifestWithoutInputs(t *testing.T) {
	dir := t.TempDir()
	manifest := `{"schema":"nereid-model/1","model":{"id":"chat","version":"1.0.0","task":"chat"},
"runtime":{"backend":"openai","model_name":"chat-model"}}`
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	m, _, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if m.Runtime.Backend != "openai" {
		t.Fatalf("backend = %q", m.Runtime.Backend)
	}
}

func TestLoadRejectsUnknownServerFamily(t *testing.T) {
	dir := t.TempDir()
	manifest := `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","task":"detection"},
"artifact":{"format":"onnx","precision":"FP32"},
"runtime":{"backend":"kserve","model_name":"m","server":{"family":"mystery-server"}},
"io":{"inputs":[{"name":"images","shape":[1,3,4,4],"layout":"NCHW","dtype":"FP32"}],
      "outputs":[{"name":"o","shape":[1,2],"layout":"NC","dtype":"FP32"}]},
"decoder":{"kind":"classification.probs","output":"o"}}`
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("unknown runtime.server.family must be rejected at validation")
	}
}

func TestLoadValidatesServedBySlots(t *testing.T) {
	valid := `{"schema":"nereid-model/1","model":{"id":"m","version":"1.0.0","task":"detection","labels_file":"labels.txt"},
"artifact":{"format":"onnx","precision":"FP32"},
"runtime":{"backend":"kserve","model_name":"m","server":{"family":"triton"},
  "served_by":[
    {"family":"triton","backend_hint":"onnxruntime","payload":"payloads/onnx","targets":["CPU"]},
    {"family":"ovms","payload":"repository/m"}
  ]},
"io":{"inputs":[{"name":"images","shape":[1,3,4,4],"layout":"NCHW","dtype":"FP32"}],
      "outputs":[{"name":"o","shape":[1,2],"layout":"NC","dtype":"FP32"}]},
"decoder":{"kind":"classification.probs","output":"o"}}`
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), []byte(valid), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "labels.txt"), []byte("a\nb\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	m, _, err := Load(dir)
	if err != nil {
		t.Fatalf("valid served_by table rejected: %v", err)
	}
	if len(m.Runtime.ServedBy) != 2 || m.Runtime.ServedBy[0].BackendHint != "onnxruntime" {
		t.Fatalf("served_by not parsed: %+v", m.Runtime.ServedBy)
	}

	for name, mutate := range map[string]func(string) string{
		"unknown slot family": func(s string) string {
			return strings.Replace(s, `"family":"ovms"`, `"family":"mystery"`, 1)
		},
		"traversal payload": func(s string) string {
			return strings.Replace(s, `"payload":"repository/m"`, `"payload":"../escape"`, 1)
		},
		"empty payload": func(s string) string {
			return strings.Replace(s, `"payload":"payloads/onnx"`, `"payload":" "`, 1)
		},
		"duplicate family slot": func(s string) string {
			return strings.Replace(s, `{"family":"ovms","payload":"repository/m"}`,
				`{"family":"triton","backend_hint":"onnxruntime","payload":"payloads/dupe"}`, 1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			bad := mutate(valid)
			badDir := t.TempDir()
			if err := os.WriteFile(filepath.Join(badDir, "nereid-model.json"), []byte(bad), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(badDir, "labels.txt"), []byte("a\nb\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, _, err := Load(badDir); err == nil {
				t.Fatal("invalid served_by slot must be rejected")
			}
		})
	}
}
