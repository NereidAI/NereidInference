// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"testing"

	dsModels "github.com/edgexfoundry/device-sdk-go/v4/pkg/models"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/common"

	"github.com/edgexfoundry/device-inference/internal/job"
)

func TestHandleReadCommandsExposesOptionalHealthAndPipelineVersions(t *testing.T) {
	const deviceName = "pipeline-health"
	j := job.New(deviceName, job.Config{PipelineID: "face"}, job.Deps{})
	d := &InferenceDriver{jobs: map[string]*jobState{
		deviceName: {name: deviceName, cfg: job.Config{PipelineID: "face"}, job: j},
	}}
	reqs := []dsModels.CommandRequest{
		{DeviceResourceName: "State", Type: common.ValueTypeString},
		{DeviceResourceName: "LastError", Type: common.ValueTypeString},
		{DeviceResourceName: "SuccessTotal", Type: common.ValueTypeInt64},
		{DeviceResourceName: "ErrorTotal", Type: common.ValueTypeInt64},
		{DeviceResourceName: "P50Ms", Type: common.ValueTypeFloat64},
		{DeviceResourceName: "P95Ms", Type: common.ValueTypeFloat64},
		{DeviceResourceName: ActivePipelineVersionResource, Type: common.ValueTypeString},
		{DeviceResourceName: LatestPipelineVersionResource, Type: common.ValueTypeString},
	}
	values, err := d.HandleReadCommands(deviceName, nil, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != len(reqs) {
		t.Fatalf("read returned %d values, want %d", len(values), len(reqs))
	}
	for i, value := range values {
		if value.DeviceResourceName != reqs[i].DeviceResourceName || value.Type != reqs[i].Type {
			t.Errorf("value %d has identity/type %s/%s, want %s/%s", i,
				value.DeviceResourceName, value.Type, reqs[i].DeviceResourceName, reqs[i].Type)
		}
	}
	if got := values[0].Value; got != job.StateApplying {
		t.Errorf("State = %#v, want %q", got, job.StateApplying)
	}
	if got := values[1].Value; got != "" {
		t.Errorf("LastError = %#v, want empty", got)
	}
	for _, index := range []int{2, 3} {
		if got := values[index].Value; got != int64(0) {
			t.Errorf("%s = %#v (%T), want int64(0)", reqs[index].DeviceResourceName, got, got)
		}
	}
	for _, index := range []int{4, 5} {
		if got := values[index].Value; got != float64(0) {
			t.Errorf("%s = %#v (%T), want float64(0)", reqs[index].DeviceResourceName, got, got)
		}
	}
	// Zero latency with SuccessTotal=0 is only an empty statistics reading; it
	// is not evidence that an inference has succeeded or that a runtime is well.
	for _, index := range []int{6, 7} {
		if got := values[index].Value; got != "" {
			t.Errorf("%s = %#v, want an unresolved empty version", reqs[index].DeviceResourceName, got)
		}
	}
}
