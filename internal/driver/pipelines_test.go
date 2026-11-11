// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	dsModels "github.com/edgexfoundry/device-sdk-go/v4/pkg/models"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/common"

	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/pipeline/testsupport"
)

func TestPipelinesManagerResourceRoundTripAndAtomicRejection(t *testing.T) {
	root := t.TempDir()
	registry, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	driver := &InferenceDriver{pipelines: registry}
	reqs := []dsModels.CommandRequest{{DeviceResourceName: PipelinesResource, Type: common.ValueTypeString}}

	records := readPipelineManagerTable(t, driver, reqs)
	if len(records) != 0 {
		t.Fatalf("initial manager table = %+v, want empty registry", records)
	}
	base := testsupport.FaceRecognitionDefinition()
	custom := testsupport.FaceRecognitionDefinition()
	custom.PipelineID = "custom-face"
	records = []pipeline.Definition{base, custom}
	writePipelineManagerTable(t, driver, reqs, records)
	accepted := registry.Snapshot()
	if len(accepted) != 2 {
		t.Fatalf("accepted table = %+v", accepted)
	}

	reloaded, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Snapshot(); !reflect.DeepEqual(got, accepted) {
		t.Fatalf("persisted manager table differs:\n%+v\n%+v", got, accepted)
	}

	mutated := custom
	mutated.Description = "an accepted version cannot be redefined"
	if err := driver.handleManagerWrite(reqs, []*dsModels.CommandValue{
		pipelineManagerParameter(t, []pipeline.Definition{records[0], mutated}),
	}); err == nil {
		t.Fatal("immutable manager PUT was accepted")
	}
	if got := registry.Snapshot(); !reflect.DeepEqual(got, accepted) {
		t.Fatalf("rejected immutable PUT changed visible table: %+v", got)
	}
	malformed, err := dsModels.NewCommandValue(PipelinesResource, common.ValueTypeString, "{")
	if err != nil {
		t.Fatal(err)
	}
	if err := driver.handleManagerWrite(reqs, []*dsModels.CommandValue{malformed}); err == nil {
		t.Fatal("malformed manager JSON was accepted")
	}
	if got := registry.Snapshot(); !reflect.DeepEqual(got, accepted) {
		t.Fatalf("rejected malformed PUT changed visible table: %+v", got)
	}

	writePipelineManagerTable(t, driver, reqs, []pipeline.Definition{})
	if got := readPipelineManagerValue(t, driver, reqs); got != "[]" {
		t.Fatalf("empty manager table = %q, want []", got)
	}
	restarted, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if err := restarted.Replace([]pipeline.Definition{mutated}); err == nil {
		t.Fatal("empty-table replacement erased immutable version history")
	}
}

func TestPipelinesManagerRejectsRawJSONWithoutSideEffects(t *testing.T) {
	root := t.TempDir()
	registry, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	valid := testsupport.FaceRecognitionDefinition()
	if err := registry.Replace([]pipeline.Definition{valid}); err != nil {
		t.Fatal(err)
	}
	driver := &InferenceDriver{pipelines: registry}
	reqs := []dsModels.CommandRequest{{DeviceResourceName: PipelinesResource, Type: common.ValueTypeString}}
	beforeGET := readPipelineManagerValue(t, driver, reqs)
	store := filepath.Join(root, "runtime", "pipelines", "pipelines.json")
	beforeDisk, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}
	validObject, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	mutate := func(change func(map[string]any)) string {
		var object map[string]any
		if err := json.Unmarshal(validObject, &object); err != nil {
			t.Fatal(err)
		}
		change(object)
		raw, err := json.Marshal([]map[string]any{object})
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	duplicate := valid
	trimmedDuplicate := testsupport.FaceRecognitionDefinition()
	trimmedDuplicate.PipelineID = "  " + valid.PipelineID + "  "
	trimmedDuplicate.Version = "  " + valid.Version + "  "
	duplicateRaw, err := json.Marshal([]pipeline.Definition{duplicate, trimmedDuplicate})
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string]string{
		"malformed":           "{",
		"null":                "null",
		"object":              "{}",
		"scalar":              "1",
		"string":              `"table"`,
		"trailing value":      string(validObject) + " []",
		"trailing array":      "[" + string(validObject) + "] []",
		"duplicate identity":  string(duplicateRaw),
		"numeric version":     "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":1`, 1) + "]",
		"boolean version":     "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":true`, 1) + "]",
		"object version":      "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":{}`, 1) + "]",
		"array version":       "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":[]`, 1) + "]",
		"null version":        "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":null`, 1) + "]",
		"unsafe pipeline id":  "[" + strings.Replace(string(validObject), `"pipelineId":"face-recognition-v1"`, `"pipelineId":"../face"`, 1) + "]",
		"unsafe version":      "[" + strings.Replace(string(validObject), `"version":"1.0.0"`, `"version":"../1"`, 1) + "]",
		"unknown definition":  mutate(func(object map[string]any) { object["unknown"] = true }),
		"wrong-case identity": mutate(func(object map[string]any) { object["PipelineId"] = object["pipelineId"]; delete(object, "pipelineId") }),
		"wrong-case version":  mutate(func(object map[string]any) { object["Version"] = object["version"]; delete(object, "version") }),
		"unknown node field":  mutate(func(object map[string]any) { object["nodes"].([]any)[0].(map[string]any)["unknown"] = true }),
		"unknown model field": mutate(func(object map[string]any) {
			object["nodes"].([]any)[1].(map[string]any)["model"].(map[string]any)["unknown"] = true
		}),
		"unknown input field": mutate(func(object map[string]any) {
			object["nodes"].([]any)[1].(map[string]any)["inputs"].([]any)[0].(map[string]any)["unknown"] = true
		}),
		"unknown limits field":  mutate(func(object map[string]any) { object["limits"].(map[string]any)["unknown"] = true }),
		"null pipeline element": "[null]",
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			parameter, err := dsModels.NewCommandValue(PipelinesResource, common.ValueTypeString, raw)
			if err != nil {
				t.Fatal(err)
			}
			if err := driver.handleManagerWrite(reqs, []*dsModels.CommandValue{parameter}); err == nil {
				t.Fatalf("invalid raw table was accepted: %s", raw)
			}
			if got := readPipelineManagerValue(t, driver, reqs); got != beforeGET {
				t.Fatalf("rejected table changed GET:\nbefore %s\nafter  %s", beforeGET, got)
			}
			afterDisk, err := os.ReadFile(store)
			if err != nil {
				t.Fatal(err)
			}
			if string(afterDisk) != string(beforeDisk) {
				t.Fatal("rejected table changed persistent bytes")
			}
		})
	}
}

func TestPipelinesManagerCanonicalReadbackIsIdempotent(t *testing.T) {
	root := t.TempDir()
	registry, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	driver := &InferenceDriver{pipelines: registry}
	reqs := []dsModels.CommandRequest{{DeviceResourceName: PipelinesResource, Type: common.ValueTypeString}}
	unchanged := testsupport.FaceRecognitionDefinition()
	later := testsupport.FaceRecognitionDefinition()
	later.PipelineID = "  z-face  "
	later.Version = " 1.10.0 "
	later.Nodes[1].Model.ModelID = " yunet "
	later.Nodes[1].Model.ModelVersion = " 1.0.0 "
	later.Nodes[1].Model.TargetDevice = " gpu.0 "
	later.Nodes[3].Model.ModelID = " sface "
	later.Nodes[3].Model.TargetDevice = "   "
	later.Nodes[2].Mode = ""
	later.Limits = pipeline.Limits{}
	earlier := testsupport.FaceRecognitionDefinition()
	earlier.PipelineID = "a-face"
	earlier.Version = "1.01.0"
	writePipelineManagerTable(t, driver, reqs, []pipeline.Definition{later, earlier, unchanged})

	first := readPipelineManagerValue(t, driver, reqs)
	records := readPipelineManagerTable(t, driver, reqs)
	if len(records) != 3 || records[0].PipelineID != "a-face" || records[1].PipelineID != unchanged.PipelineID || records[2].PipelineID != "z-face" {
		t.Fatalf("unstable canonical table order: %+v", records)
	}
	if !reflect.DeepEqual(records[1], unchanged) {
		t.Fatalf("normalizing one entry changed a non-target entry:\n%+v\n%+v", records[1], unchanged)
	}
	got := records[2]
	if got.Version != "1.10.0" || got.Nodes[1].Model.ModelID != "yunet" || got.Nodes[1].Model.ModelVersion != "1.0.0" ||
		got.Nodes[1].Model.TargetDevice != "GPU.0" || got.Nodes[3].Model.TargetDevice != "" ||
		got.Nodes[2].Mode != "five-point" || got.Limits.MaxObjects != 8 || got.Limits.MaxEventBytes != 64*1024 {
		t.Fatalf("PUT did not return the canonical definition: %+v", got)
	}
	parameter, err := dsModels.NewCommandValue(PipelinesResource, common.ValueTypeString, first)
	if err != nil {
		t.Fatal(err)
	}
	if err := driver.handleManagerWrite(reqs, []*dsModels.CommandValue{parameter}); err != nil {
		t.Fatalf("canonical GET replay failed: %v", err)
	}
	if second := readPipelineManagerValue(t, driver, reqs); second != first {
		t.Fatalf("GET -> PUT -> GET is not idempotent:\n%s\n%s", first, second)
	}
	reloaded, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	restartedDriver := &InferenceDriver{pipelines: reloaded}
	if restarted := readPipelineManagerValue(t, restartedDriver, reqs); restarted != first {
		t.Fatalf("restart changed canonical readback:\n%s\n%s", first, restarted)
	}
}

func readPipelineManagerTable(t *testing.T, driver *InferenceDriver, reqs []dsModels.CommandRequest) []pipeline.Definition {
	t.Helper()
	raw := readPipelineManagerValue(t, driver, reqs)
	var records []pipeline.Definition
	if err := json.Unmarshal([]byte(raw), &records); err != nil {
		t.Fatalf("manager returned invalid pipeline JSON %q: %v", raw, err)
	}
	return records
}

func readPipelineManagerValue(t *testing.T, driver *InferenceDriver, reqs []dsModels.CommandRequest) string {
	t.Helper()
	values, err := driver.handleManagerRead(reqs)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 {
		t.Fatalf("manager GET values = %d, want 1", len(values))
	}
	raw, err := values[0].StringValue()
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func writePipelineManagerTable(t *testing.T, driver *InferenceDriver, reqs []dsModels.CommandRequest, records []pipeline.Definition) {
	t.Helper()
	if err := driver.handleManagerWrite(reqs, []*dsModels.CommandValue{pipelineManagerParameter(t, records)}); err != nil {
		t.Fatal(err)
	}
}

func pipelineManagerParameter(t *testing.T, records []pipeline.Definition) *dsModels.CommandValue {
	t.Helper()
	raw, err := json.Marshal(records)
	if err != nil {
		t.Fatal(err)
	}
	parameter, err := dsModels.NewCommandValue(PipelinesResource, common.ValueTypeString, string(raw))
	if err != nil {
		t.Fatal(err)
	}
	return parameter
}
