// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package pipeline

// faceRecognitionID and faceRecognitionDefinition are the white-box test
// fixture for the face-recognition sample graph. The production binary ships
// no business pipeline: the canonical application material is the sample
// under samples/pipelines/, and external-package tests use the parallel
// internal/pipeline/testsupport fixture. TestSampleProvisionFile... asserts
// the sample YAML matches this fixture; a driver-package test asserts the
// same sample matches testsupport, keeping all three copies equal.
const faceRecognitionID = "face-recognition-v1"

func faceRecognitionDefinition() Definition {
	return Definition{
		PipelineID:  faceRecognitionID,
		Version:     "1.0.0",
		Description: "YuNet detection and five keypoints -> ROI five-point alignment -> per-face SFace 128-D embedding",
		Nodes: []Node{
			{ID: "source", Kind: "source"},
			{ID: "detect", Kind: "infer", Model: ModelReference{ModelID: "yunet"}, Inputs: []InputBinding{{From: "frame"}}},
			{ID: "align", Kind: "transform", Operator: "roi-align-crop", Mode: "five-point"},
			{ID: "embed", Kind: "infer", Model: ModelReference{ModelID: "sface"}, Inputs: []InputBinding{{From: "crops"}}},
			{ID: "emit", Kind: "emit"},
		},
		Edges:  [][2]string{{"source", "detect"}, {"detect", "align"}, {"align", "embed"}, {"embed", "emit"}},
		Limits: Limits{MaxObjects: 8, MaxEventBytes: 64 * 1024},
	}
}
