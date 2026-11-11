// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

type testServiceConfiguration struct {
	Service struct {
		Port int `yaml:"Port"`
	} `yaml:"Service"`
	Device struct {
		EnableAsyncReadings bool `yaml:"EnableAsyncReadings"`
		AsyncBufferSize     int  `yaml:"AsyncBufferSize"`
	} `yaml:"Device"`
	Inference InferenceConfig `yaml:"Inference"`
}

func TestProvidedConfiguration(t *testing.T) {
	path := filepath.Join("..", "..", "cmd", "res", "configuration.yaml")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg testServiceConfiguration
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Service.Port != 59992 {
		t.Fatalf("service port = %d, want 59992", cfg.Service.Port)
	}
	if !cfg.Device.EnableAsyncReadings || cfg.Device.AsyncBufferSize <= 0 {
		t.Fatalf("async readings not enabled: %+v", cfg.Device)
	}
	inf := cfg.Inference
	if inf.DefaultEndpoint == "" || inf.OpenAIEndpoint == "" || inf.ModelRoot == "" {
		t.Fatalf("unexpected endpoints/model root: %+v", inf)
	}
	if !inf.Ess.Enabled || inf.Ess.LocalDir == "" || !inf.Inbox.Enabled || inf.Inbox.Dir == "" {
		t.Fatalf("model acquisition sources not configured: %+v", inf)
	}
	if !inf.Preview.Enabled || inf.Preview.Bind != "0.0.0.0:18080" {
		t.Fatalf("unexpected preview config: %+v", inf.Preview)
	}
	if inf.FFmpeg.Bin != "ffmpeg" || inf.FFmpeg.ProbeBin != "ffprobe" ||
		inf.FFmpeg.DropUID != 1000 || inf.FFmpeg.OutputTemplate == "" {
		t.Fatalf("unexpected ffmpeg config: %+v", inf.FFmpeg)
	}
	if !inf.FFmpeg.AuthChallengeNormalize {
		t.Fatal("AuthChallengeNormalize must default on in provided configuration")
	}
	if inf.FFmpeg.VAAPIDevice != "/dev/dri/renderD128" {
		t.Fatalf("VAAPIDevice = %q, want /dev/dri/renderD128", inf.FFmpeg.VAAPIDevice)
	}
	if !inf.Onvif.NormalizeVirtualChannel || inf.Onvif.MainChannel != "101" {
		t.Fatalf("unexpected onvif config: %+v", inf.Onvif)
	}
	if !inf.valid() {
		t.Fatal("provided inference configuration is invalid")
	}
}

func TestValidateRejectsBadLimitsAndPrivileges(t *testing.T) {
	base := New()
	base.ModelRoot = t.TempDir()
	for name, mutate := range map[string]func(*InferenceConfig){
		"max-jobs":        func(c *InferenceConfig) { c.Limits.MaxJobs = -1 },
		"frame-ttl":       func(c *InferenceConfig) { c.Limits.FrameTTLFactor = 0 },
		"json-cap":        func(c *InferenceConfig) { c.Limits.JSONPayloadCap = 0 },
		"timeout":         func(c *InferenceConfig) { c.FFmpeg.ConnectTimeoutSec = 0 },
		"drop-pair":       func(c *InferenceConfig) { c.FFmpeg.DropGID = 0 },
		"drop-negative":   func(c *InferenceConfig) { c.FFmpeg.DropUID = -1; c.FFmpeg.DropGID = -1 },
		"drop-overflow":   func(c *InferenceConfig) { c.FFmpeg.DropUID = 4294967296; c.FFmpeg.DropGID = 4294967296 },
		"timeout-too-big": func(c *InferenceConfig) { c.FFmpeg.ConnectTimeoutSec = 61 },
		"vaapi-relative":  func(c *InferenceConfig) { c.FFmpeg.VAAPIDevice = "dri/renderD128" },
		"main-channel-sep": func(c *InferenceConfig) {
			c.Onvif.NormalizeVirtualChannel = true
			c.Onvif.MainChannel = "../101"
		},
		"main-channel-empty": func(c *InferenceConfig) {
			c.Onvif.NormalizeVirtualChannel = true
			c.Onvif.MainChannel = " "
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := base
			mutate(&cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("expected invalid configuration")
			}
		})
	}
}
