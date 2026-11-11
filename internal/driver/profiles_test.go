// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/edgexfoundry/go-mod-core-contracts/v4/common"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/dtos"
	"gopkg.in/yaml.v3"
)

type expectedResource struct {
	valueType string
	readWrite string
}

func TestInferenceResourceTypesAndAccess(t *testing.T) {
	health := map[string]expectedResource{
		"State":        {common.ValueTypeString, "R"},
		"LastError":    {common.ValueTypeString, "R"},
		"SuccessTotal": {common.ValueTypeInt64, "R"},
		"ErrorTotal":   {common.ValueTypeInt64, "R"},
		"P50Ms":        {common.ValueTypeFloat64, "R"},
		"P95Ms":        {common.ValueTypeFloat64, "R"},
	}
	for profileName, expected := range map[string]map[string]expectedResource{
		"device.inference.detection.yaml": {
			"DetectionJSON":  {common.ValueTypeString, "R"},
			"DetectionCount": {common.ValueTypeInt32, "R"},
			"Top1Label":      {common.ValueTypeString, "R"},
			"Top1Score":      {common.ValueTypeFloat64, "R"},
			"FrameJPEG":      {common.ValueTypeBinary, "R"},
			"InferLatencyMs": {common.ValueTypeFloat64, "R"},
			"Trigger":        {common.ValueTypeString, "W"},
		},
		"device.inference.pipeline.yaml": {
			"PipelineJSON":          {common.ValueTypeString, "R"},
			"ActivePipelineVersion": {common.ValueTypeString, "R"},
			"LatestPipelineVersion": {common.ValueTypeString, "R"},
			"InferLatencyMs":        {common.ValueTypeFloat64, "R"},
			"Trigger":               {common.ValueTypeString, "W"},
		},
	} {
		t.Run(profileName, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join("..", "..", "cmd", "res", "profiles", profileName))
			if err != nil {
				t.Fatal(err)
			}
			var profile dtos.DeviceProfile
			if err := yaml.Unmarshal(b, &profile); err != nil {
				t.Fatal(err)
			}
			actual := map[string]expectedResource{}
			for _, resource := range profile.DeviceResources {
				actual[resource.Name] = expectedResource{resource.Properties.ValueType, resource.Properties.ReadWrite}
			}
			for name, want := range health {
				if got := actual[name]; got != want {
					t.Errorf("resource %s = %+v, want %+v", name, got, want)
				}
			}
			for name, want := range expected {
				if got := actual[name]; got != want {
					t.Errorf("resource %s = %+v, want %+v", name, got, want)
				}
			}
		})
	}
}

func TestManagerResourceAccess(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "cmd", "res", "profiles", "device.inference.manager.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var profile dtos.DeviceProfile
	if err := yaml.Unmarshal(b, &profile); err != nil {
		t.Fatal(err)
	}
	actual := map[string]expectedResource{}
	for _, resource := range profile.DeviceResources {
		actual[resource.Name] = expectedResource{resource.Properties.ValueType, resource.Properties.ReadWrite}
	}
	for name, want := range map[string]expectedResource{
		ModelsResource:            {common.ValueTypeString, "R"},
		ObjectDiagnosticsResource: {common.ValueTypeString, "R"},
		ExternalServablesResource: {common.ValueTypeString, "R"},
		PipelinesResource:         {common.ValueTypeString, "RW"},
	} {
		if got := actual[name]; got != want {
			t.Errorf("manager resource %s = %+v, want %+v", name, got, want)
		}
	}
}

func TestProvidedProfilesParse(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("..", "..", "cmd", "res", "profiles", "device.inference.*.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 13 {
		t.Fatalf("expected 12 job profiles plus 1 manager profile, got %d: %v", len(files), files)
	}
	for _, path := range files {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		var profile dtos.DeviceProfile
		if err := yaml.Unmarshal(b, &profile); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if profile.Name == "" || len(profile.DeviceResources) == 0 {
			t.Fatalf("%s: missing profile name or resources", path)
		}
		seen := map[string]bool{}
		var hasTrigger bool
		for _, resource := range profile.DeviceResources {
			if resource.Name == "" || seen[resource.Name] {
				t.Fatalf("%s: invalid/duplicate resource %q", path, resource.Name)
			}
			seen[resource.Name] = true
			if resource.Properties.ValueType == "" {
				t.Fatalf("%s: resource %s lacks valueType", path, resource.Name)
			}
			switch resource.Properties.ValueType {
			case common.ValueTypeString, common.ValueTypeInt32, common.ValueTypeInt64,
				common.ValueTypeFloat64, common.ValueTypeBinary, common.ValueTypeBool:
			default:
				t.Fatalf("%s: resource %s has unsupported valueType %q", path, resource.Name, resource.Properties.ValueType)
			}
			if resource.Name == "Trigger" {
				hasTrigger = true
				if resource.Properties.ReadWrite != "W" {
					t.Fatalf("%s: Trigger must be write-only, got %q", path, resource.Properties.ReadWrite)
				}
			}
		}
		if profile.Name == ManagerProfileName {
			for _, required := range []string{ModelsResource, ObjectDiagnosticsResource, ExternalServablesResource, PipelinesResource} {
				if !seen[required] {
					t.Fatalf("%s: missing manager resource %s", path, required)
				}
			}
			continue
		}
		for _, required := range []string{"State", "InferLatencyMs", "SuccessTotal", "ErrorTotal"} {
			if !seen[required] {
				t.Fatalf("%s: missing required resource %s", path, required)
			}
		}
		if !hasTrigger {
			t.Fatalf("%s: missing Trigger resource", path)
		}
	}
}
