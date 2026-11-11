// -*- Mode: Go; indent-tabs-mode: t -*-
//
// Copyright (C) 2026 YIQISOFT
//
// SPDX-License-Identifier: Apache-2.0

// Package device_inference holds the global build version of the service.
// The value is injected at build time by the Makefile via -ldflags.
package device_inference

// Version is replaced at link time; see the Makefile GOFLAGS definition. The
// fallback is the rolling dev channel; release builds overwrite it with bare
// SemVer derived from the git tag (v0.0.1 -> 0.0.1).
var Version string = "0.0.0-dev"
