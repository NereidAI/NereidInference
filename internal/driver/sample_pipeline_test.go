// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/pipeline/testsupport"
)

// TestSamplePipelineMatchesTestsupportFixture guards the single source of
// truth for the face-recognition sample: the importable YAML under
// samples/pipelines/, the external-package testsupport fixture and (via the
// pipeline package's own sample test) its white-box fixture must all encode
// the same definition. Drift in any copy fails here or in the pipeline
// package test.
func TestSamplePipelineMatchesTestsupportFixture(t *testing.T) {
	registry, err := pipeline.NewRegistry(pipeline.RegistryConfig{
		ModelRoot:     t.TempDir(),
		ProvisionFile: filepath.Join("..", "..", "samples", "pipelines", "face-recognition-v1.yaml"),
	})
	if err != nil {
		t.Fatal(err)
	}
	provisioned, ok := registry.Lookup(testsupport.FaceRecognitionID)
	if !ok {
		t.Fatal("sample pipeline was not registered from the provision file")
	}
	if !reflect.DeepEqual(provisioned, testsupport.FaceRecognitionDefinition()) {
		t.Fatalf("sample YAML and testsupport fixture diverge:\n%#v\n%#v",
			provisioned, testsupport.FaceRecognitionDefinition())
	}
}
