// -*- Mode: Go; indent-tabs-mode: t -*-
//
// Copyright (C) 2017-2018 Canonical Ltd
// Copyright (C) 2018-2019 IOTech Ltd
// Copyright (C) 2026 YIQISOFT
//
// SPDX-License-Identifier: Apache-2.0
//
// Based on EdgeX device-sdk-go example/cmd/device-simple/main.go and modified
// for NereidInference.

// Command device-inference is the EdgeX device service entry point. It wires
// the generic inference protocol driver into the EdgeX device SDK bootstrap.
package main

import (
	"github.com/edgexfoundry/device-sdk-go/v4/pkg/startup"

	"github.com/edgexfoundry/device-inference"
	"github.com/edgexfoundry/device-inference/internal/driver"
)

// serviceName is the EdgeX service key; it must match the configuration
// Service.Name used by core-metadata and the compose deployment.
const serviceName string = "device-inference"

func main() {
	d := driver.NewInferenceDriver()
	startup.Bootstrap(serviceName, device_inference.Version, d)
}
