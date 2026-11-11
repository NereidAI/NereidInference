// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"strings"
	"testing"
)

func TestServableIDAdmissionFailsClosed(t *testing.T) {
	for name, props := range map[string]map[string]any{
		"only":                  {"ServableID": "remote-face", "SourceKind": "none"},
		"with pipeline":         {"ServableID": " remote-face ", "PipelineId": "face", "SourceKind": "jpeg", "SourceURL": "http://camera/snapshot"},
		"with local model":      {"ServableID": "remote-face", "ModelID": "local", "SourceKind": "none"},
		"with both selectors":   {"ServableID": "remote-face", "PipelineId": "face", "ModelID": "local", "SourceKind": "none"},
		"with explicit CPU":     {"ServableID": "remote-face", "ModelID": "local", "SourceKind": "none", "TargetDevice": "CPU"},
		"with explicit GPU":     {"ServableID": "remote-face", "ModelID": "local", "SourceKind": "none", "TargetDevice": "gpu.0"},
		"with malformed target": {"ServableID": "remote-face", "ModelID": "local", "SourceKind": "none", "TargetDevice": "bad;target"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := FromProperties(props)
			if err == nil || !strings.Contains(err.Error(), ReasonConfigInvalid) || !strings.Contains(err.Error(), "Inference.ServableID") {
				t.Fatalf("ServableID admission error = %v, want config-invalid naming the exact property", err)
			}
		})
	}
	for _, props := range []map[string]any{
		{"ModelID": "local", "SourceKind": "none"},
		{"ModelID": "local", "SourceKind": "none", "ServableID": "   "},
	} {
		if _, err := FromProperties(props); err != nil {
			t.Fatalf("omitted/blank ServableID changed managed admission: %v", err)
		}
	}
}

func TestPipelineSourceKindRequirements(t *testing.T) {
	for _, test := range []struct {
		kind       string
		required   string
		properties map[string]any
	}{
		{kind: "onvif-camera", required: "SourceDevice", properties: map[string]any{"SourceDevice": "camera-1"}},
		{kind: "onvif-snapshot", required: "SourceDevice", properties: map[string]any{"SourceDevice": "camera-1"}},
		{kind: "rtsp", required: "SourceURL", properties: map[string]any{"SourceURL": "rtsp://camera/live"}},
		{kind: "rtsp-udp", required: "SourceURL", properties: map[string]any{"SourceURL": "rtsp://camera/live"}},
		{kind: "mpjpeg", required: "SourceURL", properties: map[string]any{"SourceURL": "http://camera/mjpeg"}},
		{kind: "jpeg", required: "SourceURL", properties: map[string]any{"SourceURL": "http://camera/snapshot"}},
	} {
		t.Run(test.kind, func(t *testing.T) {
			props := map[string]any{"PipelineId": "face", "SourceKind": test.kind, "Trigger": TriggerCommand}
			for key, value := range test.properties {
				props[key] = value
			}
			if _, err := FromProperties(props); err != nil {
				t.Fatalf("valid %s source rejected: %v", test.kind, err)
			}
			delete(props, test.required)
			if _, err := FromProperties(props); err == nil {
				t.Fatalf("%s without %s must be rejected", test.kind, test.required)
			}
		})
	}
	if _, err := FromProperties(map[string]any{"PipelineId": "face", "SourceKind": "none"}); err == nil {
		t.Fatal("pipeline source=none must be rejected")
	}
	if _, err := FromProperties(map[string]any{"ModelID": "legacy", "SourceKind": "none"}); err != nil {
		t.Fatalf("legacy command-only source=none regressed: %v", err)
	}
}

func TestPipelineTargetDeviceDefaultsAndNormalizes(t *testing.T) {
	base := func() map[string]any {
		return map[string]any{"PipelineId": "face", "SourceKind": "jpeg", "SourceURL": "http://camera/snapshot"}
	}
	props := base()
	cfg, err := FromProperties(props)
	if err != nil || cfg.TargetDevice != "CPU" {
		t.Fatalf("default TargetDevice = %q, %v; want CPU", cfg.TargetDevice, err)
	}
	props = base()
	props["TargetDevice"] = " gpu.2 "
	cfg, err = FromProperties(props)
	if err != nil || cfg.TargetDevice != "GPU.2" {
		t.Fatalf("normalized TargetDevice = %q, %v; want GPU.2", cfg.TargetDevice, err)
	}
	props = base()
	props["TargetDevice"] = " CPU; rm -rf / "
	if _, err := FromProperties(props); err == nil {
		t.Fatal("unsafe TargetDevice must be rejected")
	}
}

func TestPipelineIDSelectionIsExclusiveAndExact(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"PipelineId": "face-recognition-v1", "SourceKind": "jpeg", "SourceURL": "http://photo/one.jpg",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.PipelineID != "face-recognition-v1" || cfg.ModelID != "" {
		t.Fatalf("unexpected pipeline config: %+v", cfg)
	}
	if _, err := FromProperties(map[string]any{
		"ModelID": "legacy", "SourceKind": "jpeg", "SourceURL": "http://camera/snap",
	}); err != nil {
		t.Fatalf("legacy single-model selection regressed: %v", err)
	}

	for name, props := range map[string]map[string]any{
		"both":                    {"PipelineId": "face", "ModelID": "legacy", "SourceKind": "jpeg", "SourceURL": "http://photo"},
		"pipeline plus version":   {"PipelineId": "face", "ModelVersion": "1.0.0", "SourceKind": "jpeg", "SourceURL": "http://photo"},
		"missing both":            {"SourceKind": "jpeg", "SourceURL": "http://photo"},
		"wrong property spelling": {"PipelineID": "face", "SourceKind": "jpeg", "SourceURL": "http://photo"},
		"no image source":         {"PipelineId": "face", "SourceKind": "none"},
		"unsafe id":               {"PipelineId": "../face", "SourceKind": "jpeg", "SourceURL": "http://photo"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := FromProperties(props)
			if err == nil {
				t.Fatal("expected selection validation error")
			}
			if name == "both" && !strings.Contains(err.Error(), "mutually exclusive") {
				t.Fatalf("ambiguous error %q does not explain exclusivity", err)
			}
		})
	}
}
