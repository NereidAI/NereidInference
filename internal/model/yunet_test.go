// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAcceptsDescriptorDrivenYuNetAndAlignmentTemplate(t *testing.T) {
	dir := t.TempDir()
	writeManifestMap(t, dir, yuNetManifestMap(32))
	manifest, _, err := Load(dir)
	if err != nil {
		t.Fatalf("valid YuNet package rejected: %v", err)
	}
	kind, err := DecoderKind(manifest.Decoder)
	if err != nil || kind != "detection.yunet" {
		t.Fatalf("decoder kind = %q, %v", kind, err)
	}

	faceDir := t.TempDir()
	writeManifestMap(t, faceDir, map[string]any{
		"schema":   "nereid-model/1",
		"model":    map[string]any{"id": "sface", "version": "1.0.0", "task": "embeddings"},
		"artifact": map[string]any{"format": "onnx", "precision": "FP32"},
		"runtime":  map[string]any{"backend": "kserve", "model_name": "sface-wire", "server": map[string]any{"family": "triton"}},
		"io": map[string]any{
			"inputs": []any{map[string]any{
				"name": "input", "shape": []int64{1, 3, 2, 2}, "layout": "NCHW", "dtype": "FP32", "color_order": "RGB",
				"preprocess": map[string]any{"resize": "stretch", "scale": "1/255", "alignment": map[string]any{
					"template": [][]float32{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}},
				}},
			}},
			"outputs": []any{map[string]any{"name": "embedding", "shape": []int64{1, 128}, "dtype": "FP32"}},
		},
		"decoder": map[string]any{"kind": "embeddings.vector", "output": "embedding"},
	})
	face, _, err := Load(faceDir)
	if err != nil {
		t.Fatalf("valid aligned SFace package rejected: %v", err)
	}
	if got := len(face.IO.Inputs[0].Preprocess.Alignment.Template); got != 5 {
		t.Fatalf("alignment points = %d, want 5", got)
	}
}

func TestLoadRejectsInvalidYuNetReferencesAndShapes(t *testing.T) {
	for name, mutate := range map[string]func(map[string]any){
		"unknown activation": func(m map[string]any) {
			m["decoder"].(map[string]any)["activation"] = "sigmoid"
		},
		"missing output": func(m map[string]any) {
			outputs := m["io"].(map[string]any)["outputs"].([]any)
			m["io"].(map[string]any)["outputs"] = outputs[:len(outputs)-1]
		},
		"wrong kps channels": func(m map[string]any) {
			outputs := m["io"].(map[string]any)["outputs"].([]any)
			for _, raw := range outputs {
				output := raw.(map[string]any)
				if output["name"] == "kps_8" {
					output["shape"] = []int64{1, 16, 5}
				}
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			manifest := yuNetManifestMap(32)
			mutate(manifest)
			writeManifestMap(t, dir, manifest)
			if _, _, err := Load(dir); err == nil {
				t.Fatal("invalid YuNet descriptor must be rejected at package admission")
			}
		})
	}
	t.Run("non-divisible input", func(t *testing.T) {
		dir := t.TempDir()
		writeManifestMap(t, dir, yuNetManifestMap(30))
		if _, _, err := Load(dir); err == nil {
			t.Fatal("YuNet input not divisible by all strides must be rejected")
		}
	})
}

func writeManifestMap(t *testing.T, dir string, value map[string]any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "nereid-model.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

func yuNetManifestMap(input int) map[string]any {
	outputs := make([]any, 0, 12)
	strides := make([]any, 0, 3)
	for _, stride := range []int{8, 16, 32} {
		anchors := (input / stride) * (input / stride)
		strideSpec := map[string]any{"stride": stride}
		for _, role := range []struct {
			name     string
			channels int
		}{{"cls", 1}, {"obj", 1}, {"bbox", 4}, {"kps", 10}} {
			name := fmt.Sprintf("%s_%d", role.name, stride)
			strideSpec[role.name] = name
			outputs = append(outputs, map[string]any{"name": name, "shape": []int64{1, int64(anchors), int64(role.channels)}, "dtype": "FP32"})
		}
		strides = append(strides, strideSpec)
	}
	return map[string]any{
		"schema":   "nereid-model/1",
		"model":    map[string]any{"id": "any-detector", "version": "1.0.0", "task": "detection"},
		"artifact": map[string]any{"format": "onnx", "precision": "FP32"},
		"runtime":  map[string]any{"backend": "kserve", "model_name": "wire-name", "server": map[string]any{"family": "triton"}},
		"io": map[string]any{
			"inputs":  []any{map[string]any{"name": "input", "shape": []int64{1, 3, int64(input), int64(input)}, "layout": "NCHW", "dtype": "FP32"}},
			"outputs": outputs,
		},
		"decoder": map[string]any{"kind": "detection.yunet", "threshold": 0.6, "strides": strides},
	}
}
