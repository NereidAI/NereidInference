// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package testsupport provides shared pipeline fixtures for NI tests. It is
// imported by _test.go files only and must never be referenced by production
// code: business-specific graph definitions (face recognition today, any
// other application tomorrow) live in importable samples under samples/, not
// in the platform binary. See the Platform Kernel Closure treaty.
package testsupport

import "github.com/edgexfoundry/device-inference/internal/pipeline"

// FaceRecognitionID is the pipeline ID of the face-recognition sample graph.
const FaceRecognitionID = "face-recognition-v1"

// FaceRecognitionDefinition returns a fresh copy of the face-recognition
// sample graph. Model versions float so an installed NF package resolves at
// bind time; tensor names, preprocessing and alignment geometry remain
// descriptor-owned. Callers may mutate the returned definition freely.
func FaceRecognitionDefinition() pipeline.Definition {
	return pipeline.Definition{
		PipelineID:  FaceRecognitionID,
		Version:     "1.0.0",
		Description: "YuNet detection and five keypoints -> ROI five-point alignment -> per-face SFace 128-D embedding",
		Nodes: []pipeline.Node{
			{ID: "source", Kind: "source"},
			{ID: "detect", Kind: "infer", Model: pipeline.ModelReference{ModelID: "yunet"}, Inputs: []pipeline.InputBinding{{From: "frame"}}},
			{ID: "align", Kind: "transform", Operator: "roi-align-crop", Mode: "five-point"},
			{ID: "embed", Kind: "infer", Model: pipeline.ModelReference{ModelID: "sface"}, Inputs: []pipeline.InputBinding{{From: "crops"}}},
			{ID: "emit", Kind: "emit"},
		},
		Edges:  [][2]string{{"source", "detect"}, {"detect", "align"}, {"align", "embed"}, {"embed", "emit"}},
		Limits: pipeline.Limits{MaxObjects: 8, MaxEventBytes: 64 * 1024},
	}
}
