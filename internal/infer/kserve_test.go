// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package infer

import (
	"encoding/binary"
	"math"
	"testing"

	pb "github.com/edgexfoundry/device-inference/internal/infer/kserve/grpcv2"
)

func TestParseKServeRawSignedTensors(t *testing.T) {
	raw8 := []byte{0, 255}
	raw32 := make([]byte, 8)
	binary.LittleEndian.PutUint32(raw32[0:], math.Float32bits(1.5))
	binary.LittleEndian.PutUint32(raw32[4:], math.Float32bits(-2.25))
	resp := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{
			{Name: "i8", Datatype: "INT8", Shape: []int64{1, 2}},
			{Name: "fp32", Datatype: "FP32", Shape: []int64{1, 2}},
		},
		RawOutputContents: [][]byte{raw8, raw32},
	}
	got, err := parseKServeResponse(resp)
	if err != nil {
		t.Fatal(err)
	}
	if vals := got.Outputs["i8"].Float32; vals[0] != 0 || vals[1] != -1 {
		t.Fatalf("INT8 values = %v", vals)
	}
	if vals := got.Outputs["fp32"].Float32; vals[0] != 1.5 || vals[1] != -2.25 {
		t.Fatalf("FP32 values = %v", vals)
	}
}

func TestParseKServeRejectsShapeMismatch(t *testing.T) {
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint32(raw, math.Float32bits(1))
	resp := &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name: "fp32", Datatype: "FP32", Shape: []int64{1, 2},
		}},
		RawOutputContents: [][]byte{raw},
	}
	if _, err := parseKServeResponse(resp); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestShapeElementsRejectsOverflow(t *testing.T) {
	if n, ok := shapeElements([]int64{1 << 35, 1 << 35}); ok || n != 0 {
		t.Fatalf("shapeElements overflow = %d,%v", n, ok)
	}
	if n, ok := shapeElements([]int64{2, 3}); !ok || n != 6 {
		t.Fatalf("shapeElements valid shape = %d,%v", n, ok)
	}
}

func TestParseKServeTypedInt64(t *testing.T) {
	resp := &pb.ModelInferResponse{Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
		Name: "labels", Datatype: "INT64", Shape: []int64{2},
		Contents: &pb.InferTensorContents{Int64Contents: []int64{3, 7}},
	}}}
	got, err := parseKServeResponse(resp)
	if err != nil {
		t.Fatal(err)
	}
	if vals := got.Outputs["labels"].Float32; vals[0] != 3 || vals[1] != 7 {
		t.Fatalf("INT64 values = %v", vals)
	}
}

func TestNewOpenAIBackendValidatesURL(t *testing.T) {
	for _, raw := range []string{"not-a-url", "file:///tmp/model"} {
		if _, err := NewOpenAIBackend(raw, ""); err == nil {
			t.Fatalf("expected %q to be rejected", raw)
		}
	}
	if _, err := NewOpenAIBackend("http://127.0.0.1:0", ""); err != nil {
		t.Fatal(err)
	}
}
