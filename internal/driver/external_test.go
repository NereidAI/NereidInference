// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	dsModels "github.com/edgexfoundry/device-sdk-go/v4/pkg/models"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/common"

	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/pipeline/testsupport"
)

func TestExternalServablesMergedReadOnlyAndMixedWritePreflight(t *testing.T) {
	root := t.TempDir()
	provision := filepath.Join(t.TempDir(), "external.yaml")
	if err := os.WriteFile(provision, []byte(`servables:
  - servableId: provisioned-one
    family: external
    protocol: kserve-grpc
    endpoint: provisioned:8001
    modelName: model_a
`), 0o600); err != nil {
		t.Fatal(err)
	}
	legacy := []modelserver.ServableRecord{{
		ServableID: "legacy-one", Family: "external", Protocol: "kserve-grpc",
		Endpoint: "legacy:8001", ModelName: "model_b",
	}}
	legacyRaw, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	legacyRaw = append(legacyRaw, '\n')
	storeDir := filepath.Join(root, "runtime", "external")
	if err := os.MkdirAll(storeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	store := filepath.Join(storeDir, "servables.json")
	if err := os.WriteFile(store, legacyRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	external, err := modelserver.NewExternal(modelserver.ExternalConfig{ModelRoot: root, ProvisionFile: provision})
	if err != nil {
		t.Fatal(err)
	}
	pipelines, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	d := &InferenceDriver{externalServables: external, pipelines: pipelines}
	req := dsModels.CommandRequest{DeviceResourceName: ExternalServablesResource, Type: common.ValueTypeString}
	values, err := d.HandleReadCommands(ManagerDeviceName, nil, []dsModels.CommandRequest{req})
	if err != nil {
		t.Fatal(err)
	}
	merged, err := values[0].StringValue()
	if err != nil {
		t.Fatal(err)
	}
	var statuses []modelserver.ServableStatus
	if err := json.Unmarshal([]byte(merged), &statuses); err != nil {
		t.Fatal(err)
	}
	if len(statuses) != 2 || statuses[0].Channel != "runtime" || statuses[1].Channel != "provisioned" {
		t.Fatalf("unexpected merged provisioned/legacy-runtime view: %+v", statuses)
	}

	parameter, err := dsModels.NewCommandValue(ExternalServablesResource, common.ValueTypeString, merged)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.HandleWriteCommands(ManagerDeviceName, nil, []dsModels.CommandRequest{req}, []*dsModels.CommandValue{parameter}); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("merged GET -> PUT error = %v, want stable read-only sentinel", err)
	}
	// Request-only, malformed, wrong-type and disabled-registry writes all have
	// the same error and none reaches JSON conversion or registry mutation.
	if err := d.HandleWriteCommands(ManagerDeviceName, nil, []dsModels.CommandRequest{req}, nil); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("parameterless external PUT error = %v", err)
	}
	malformed, _ := dsModels.NewCommandValue(ExternalServablesResource, common.ValueTypeString, "{")
	if err := d.handleManagerWrite([]dsModels.CommandRequest{req}, []*dsModels.CommandValue{malformed}); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("malformed external PUT error = %v", err)
	}
	wrongType, _ := dsModels.NewCommandValue(ExternalServablesResource, common.ValueTypeInt32, int32(1))
	if err := d.handleManagerWrite([]dsModels.CommandRequest{req}, []*dsModels.CommandValue{wrongType}); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("wrong-type external PUT error = %v", err)
	}
	disabled := &InferenceDriver{}
	if err := disabled.handleManagerWrite([]dsModels.CommandRequest{req}, []*dsModels.CommandValue{parameter}); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("disabled-registry external PUT error = %v", err)
	}

	values, err = d.HandleReadCommands(ManagerDeviceName, nil, []dsModels.CommandRequest{req})
	if err != nil {
		t.Fatal(err)
	}
	after, _ := values[0].StringValue()
	if after != merged {
		t.Fatalf("read-only PUT changed merged GET:\n%s\n%s", merged, after)
	}
	afterDisk, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}
	if string(afterDisk) != string(legacyRaw) {
		t.Fatal("read-only PUT changed legacy runtime bytes")
	}
	var persisted []modelserver.ServableRecord
	if err := json.Unmarshal(afterDisk, &persisted); err != nil {
		t.Fatal(err)
	}
	if len(persisted) != 1 || persisted[0].ServableID != "legacy-one" {
		t.Fatalf("provisioned/merged records were converted to runtime state: %+v", persisted)
	}

	// A mixed request is rejected before its first pipeline parameter can apply.
	custom := testsupport.FaceRecognitionDefinition()
	custom.PipelineID = "must-not-apply"
	pipelineParameter := pipelineManagerParameter(t, []pipeline.Definition{custom})
	mixedReqs := []dsModels.CommandRequest{
		{DeviceResourceName: PipelinesResource, Type: common.ValueTypeString}, req,
	}
	if err := d.handleManagerWrite(mixedReqs, []*dsModels.CommandValue{pipelineParameter, parameter}); !errors.Is(err, modelserver.ErrExternalServablesReadOnly) {
		t.Fatalf("mixed manager PUT error = %v", err)
	}
	if _, ok := pipelines.Lookup("must-not-apply"); ok {
		t.Fatal("mixed external rejection allowed a preceding pipeline mutation")
	}
}
