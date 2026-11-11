// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package driver

import (
	"strings"
	"testing"

	"github.com/edgexfoundry/go-mod-core-contracts/v4/clients/logger"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/models"

	"github.com/edgexfoundry/device-inference/internal/job"
)

func TestServableIDValidateAndAddDeviceHaveNoLocalFallback(t *testing.T) {
	props := models.ProtocolProperties{
		"ServableID": "remote-face",
		"ModelID":    "local-model",
		"SourceKind": "none",
		// Deliberately omit TargetDevice: rejection must not turn the default
		// into external CPU placement.
	}
	protocols := map[string]models.ProtocolProperties{ProtocolName: props}
	d := &InferenceDriver{lc: logger.NewMockClient(), jobs: map[string]*jobState{}}
	device := models.Device{Name: "external-job", ProfileName: "nereid-inference-detection", Protocols: protocols}
	if err := d.ValidateDevice(device); err == nil || !strings.Contains(err.Error(), job.ReasonConfigInvalid) {
		t.Fatalf("ValidateDevice error = %v, want config-invalid", err)
	}
	// Catalog, backends, frames and servable managers are intentionally nil.
	// Continuing into legacy local-model execution would fail rather than pass.
	if err := d.AddDevice(device.Name, protocols, models.Unlocked); err == nil || !strings.Contains(err.Error(), job.ReasonConfigInvalid) {
		t.Fatalf("AddDevice error = %v, want config-invalid", err)
	}
	if len(d.jobs) != 0 {
		t.Fatalf("rejected ServableID created a local job: %+v", d.jobs)
	}

	// Duplicate-device admission must validate new properties before returning.
	existing := job.New("external-job", job.Config{ModelID: "existing", SourceKind: "none"}, job.Deps{})
	d.jobs[device.Name] = &jobState{name: device.Name, job: existing}
	if err := d.AddDevice(device.Name, protocols, models.Unlocked); err == nil || !strings.Contains(err.Error(), "Inference.ServableID") {
		t.Fatalf("duplicate AddDevice bypassed ServableID rejection: %v", err)
	}
	if len(d.jobs) != 1 || d.jobs[device.Name].job != existing {
		t.Fatal("duplicate rejection replaced or removed the existing job")
	}
}
