// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

type yunetTestCandidate struct {
	stride       int
	anchor       int
	cls, obj     float32
	cx, cy, w, h float32
	points       [5][2]float32
}

func TestYuNetDecodeNMSKeepsBoxLandmarkAssociationAndUnmaps(t *testing.T) {
	raw := yunetTestDescriptor(t)
	response := yunetTestResponse(32, []yunetTestCandidate{
		{stride: 8, anchor: 0, cls: 0.8, obj: 0.8, cx: 32, cy: 16, w: 32, h: 32,
			points: [5][2]float32{{8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}}},
		// Same box at the adjacent grid cell, but a higher score and different
		// landmarks. NMS must retain this complete candidate, not mix fields.
		{stride: 8, anchor: 1, cls: 0.95, obj: 0.95, cx: 32, cy: 16, w: 32, h: 32,
			points: [5][2]float32{{24, 8}, {24, 12}, {32, 16}, {40, 16}, {32, 24}}},
	})
	decoder, err := newYuNetDecoder(raw)
	if err != nil {
		t.Fatal(err)
	}
	result, err := decoder.Decode(&Input{
		Manifest: &model.Manifest{Model: model.ModelSection{ID: "arbitrary-export", Version: "9"}, Decoder: raw},
		Response: response,
		Meta:     preprocess.Meta{InputW: 32, InputH: 32, SourceW: 64, SourceH: 32, ScaleX: 2, ScaleY: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	detections := result.(*DetectionResult).Detections
	if len(detections) != 1 {
		t.Fatalf("detections = %d, want 1 after NMS", len(detections))
	}
	face := detections[0]
	if face.Label != "face" || len(face.Keypoints) != 5 {
		t.Fatalf("unexpected face: %+v", face)
	}
	if math.Abs(float64(face.Score-0.95)) > 1e-5 {
		t.Fatalf("score = %g, want 0.95", face.Score)
	}
	// The detector box is clipped at the 32px model-input right edge before
	// non-square source restoration, hence x2=32/(2*64)=0.25.
	wantBox := [4]float32{0.125, 0, 0.25, 1}
	if face.Box != wantBox {
		t.Fatalf("non-square restored box = %v, want %v", face.Box, wantBox)
	}
	if math.Abs(float64(face.Keypoints[0].X-0.1875)) > 1e-6 || math.Abs(float64(face.Keypoints[0].Y-0.25)) > 1e-6 {
		t.Fatalf("first restored landmark = %+v, want (.1875,.25)", face.Keypoints[0])
	}
	if face.Keypoints[3].X == face.Keypoints[0].X {
		t.Fatal("landmarks from the suppressed candidate were retained")
	}
}

func TestYuNetScoreCombinesClassAndObjectness(t *testing.T) {
	for _, tc := range []struct {
		name       string
		activation string
		cls, obj   float32
		want       float32
	}{
		{name: "OpenCV probabilities", cls: 0.8, obj: 0.5, want: float32(math.Sqrt(0.4))},
		{name: "explicit logits", activation: "logit", cls: 4, obj: 0, want: float32(math.Sqrt(float64(sigmoid(4) * sigmoid(0))))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := yunetTestDescriptor(t)
			if tc.activation != "" {
				var spec map[string]any
				if err := json.Unmarshal(raw, &spec); err != nil {
					t.Fatal(err)
				}
				spec["activation"] = tc.activation
				var err error
				raw, err = json.Marshal(spec)
				if err != nil {
					t.Fatal(err)
				}
			}
			response := yunetTestResponse(32, []yunetTestCandidate{{
				stride: 32, anchor: 0, cls: tc.cls, obj: tc.obj, cx: 16, cy: 16, w: 16, h: 16,
				points: [5][2]float32{{8, 8}, {12, 8}, {16, 12}, {12, 16}, {20, 16}},
			}})
			decoder, err := newYuNetDecoder(raw)
			if err != nil {
				t.Fatal(err)
			}
			result, err := decoder.Decode(&Input{
				Manifest: &model.Manifest{Decoder: raw}, Response: response,
				Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 32, SourceH: 32, ScaleX: 1, ScaleY: 1},
			})
			if err != nil {
				t.Fatal(err)
			}
			got := result.(*DetectionResult).Detections[0].Score
			if math.Abs(float64(got-tc.want)) > 1e-6 {
				t.Fatalf("score = %g, want %g", got, tc.want)
			}
		})
	}
}

func TestYuNetNMSUsesUnclippedInputGeometry(t *testing.T) {
	raw := yunetTestDescriptor(t)
	response := yunetTestResponse(32, []yunetTestCandidate{
		// Raw IoU is 0.2, so both survive. Clipping both boxes to the 32px
		// input first would raise IoU to 0.5 and incorrectly suppress one.
		{stride: 8, anchor: 0, cls: 0.9, obj: 0.9, cx: 24, cy: 16, w: 16, h: 32},
		{stride: 8, anchor: 1, cls: 0.8, obj: 0.8, cx: 40, cy: 16, w: 32, h: 32},
	})
	decoder, err := newYuNetDecoder(raw)
	if err != nil {
		t.Fatal(err)
	}
	result, err := decoder.Decode(&Input{
		Manifest: &model.Manifest{Decoder: raw}, Response: response,
		Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 64, SourceH: 32, ScaleX: 2, ScaleY: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	detections := result.(*DetectionResult).Detections
	if got := len(detections); got != 2 {
		t.Fatalf("raw-geometry NMS retained %d faces, want 2: %+v", got, detections)
	}
}

func TestYuNetDropsCollapsedROIAndRetainsUnclippedAlignmentPoints(t *testing.T) {
	raw := yunetTestDescriptor(t)
	response := yunetTestResponse(32, []yunetTestCandidate{
		{stride: 8, anchor: 0, cls: 0.9, obj: 0.9, cx: 16, cy: 16, w: 16, h: 16,
			points: [5][2]float32{{40, 8}, {12, 8}, {16, 12}, {12, 16}, {20, 16}}},
		{stride: 8, anchor: 1, cls: 0.99, obj: 0.99, cx: 80, cy: 16, w: 16, h: 16},
	})
	decoder, err := newYuNetDecoder(raw)
	if err != nil {
		t.Fatal(err)
	}
	result, err := decoder.Decode(&Input{
		Manifest: &model.Manifest{Decoder: raw}, Response: response,
		Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 32, SourceH: 32, ScaleX: 1, ScaleY: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	detections := result.(*DetectionResult).Detections
	if len(detections) != 1 {
		t.Fatalf("detections = %d, want only the non-collapsed face", len(detections))
	}
	face := detections[0]
	if face.Keypoints[0].X != 1 || face.KeypointsForAlignment()[0].X <= 1 {
		t.Fatalf("border landmark clipping leaked into alignment: public=%+v alignment=%+v", face.Keypoints[0], face.KeypointsForAlignment()[0])
	}
}

func TestYuNetAcceptsNHWCAndReturnsNonNilEmptyDetections(t *testing.T) {
	raw := yunetTestDescriptor(t)
	response := yunetTestResponse(32, nil)
	for name, output := range response.Outputs {
		suffix := name[strings.LastIndex(name, "_")+1:]
		stride, err := strconv.Atoi(suffix)
		if err != nil {
			t.Fatal(err)
		}
		output.Shape = []int64{1, int64(32 / stride), int64(32 / stride), output.Shape[2]}
	}
	decoder, _ := newYuNetDecoder(raw)
	result, err := decoder.Decode(&Input{
		Manifest: &model.Manifest{Decoder: raw}, Response: response,
		Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 32, SourceH: 32, ScaleX: 1, ScaleY: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	detections := result.(*DetectionResult).Detections
	if detections == nil || len(detections) != 0 {
		t.Fatalf("healthy empty detections must be [] not nil: %#v", detections)
	}
}

func TestYuNetRejectsMissingOutputsAndMalformedShapes(t *testing.T) {
	raw := yunetTestDescriptor(t)
	for name, mutate := range map[string]func(*infer.Response){
		"missing kps": func(r *infer.Response) { delete(r.Outputs, "kps_8") },
		"wrong kps channels": func(r *infer.Response) {
			old := r.Outputs["kps_8"]
			r.Outputs["kps_8"] = &infer.TensorOutput{Name: old.Name, Shape: []int64{1, 16, 5}, Float32: make([]float32, 16*5)}
		},
		"anchor mismatch": func(r *infer.Response) {
			r.Outputs["obj_8"] = &infer.TensorOutput{Name: "obj_8", Shape: []int64{1, 15, 1}, Float32: make([]float32, 15)}
		},
		"truncated tensor": func(r *infer.Response) { r.Outputs["cls_8"].Float32 = r.Outputs["cls_8"].Float32[:15] },
		"batch two":        func(r *infer.Response) { r.Outputs["bbox_8"].Shape = []int64{2, 16, 4} },
	} {
		t.Run(name, func(t *testing.T) {
			response := yunetTestResponse(32, nil)
			mutate(response)
			decoder, _ := newYuNetDecoder(raw)
			_, err := decoder.Decode(&Input{
				Manifest: &model.Manifest{Decoder: raw}, Response: response,
				Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 32, SourceH: 32, ScaleX: 1, ScaleY: 1},
			})
			if err == nil {
				t.Fatal("expected explicit malformed output error")
			}
		})
	}
	if _, err := newYuNetDecoder(json.RawMessage(`{"kind":"detection.yunet","strides":[]}`)); err == nil {
		t.Fatal("descriptor without all stride outputs must be rejected")
	}
	if _, err := newYuNetDecoder(json.RawMessage(`{"kind":"detection.yunet","activation":"sigmoid","strides":[]}`)); err == nil {
		t.Fatal("unknown activation must be rejected")
	}
}

func TestYuNetRejectsNonFiniteCandidateData(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value float32
	}{{name: "NaN", value: float32(math.NaN())}, {name: "Inf", value: float32(math.Inf(1))}} {
		for _, role := range []string{"cls", "obj", "bbox", "kps"} {
			t.Run(tc.name+"/"+role, func(t *testing.T) {
				raw := yunetTestDescriptor(t)
				response := yunetTestResponse(32, []yunetTestCandidate{{
					stride: 8, anchor: 0, cls: 0.9, obj: 0.9, cx: 16, cy: 16, w: 16, h: 16,
					points: [5][2]float32{{12, 12}, {20, 12}, {16, 16}, {12, 20}, {20, 20}},
				}})
				response.Outputs[role+"_8"].Float32[0] = tc.value
				decoder, err := newYuNetDecoder(raw)
				if err != nil {
					t.Fatal(err)
				}
				_, err = decoder.Decode(&Input{
					Manifest: &model.Manifest{Decoder: raw}, Response: response,
					Meta: preprocess.Meta{InputW: 32, InputH: 32, SourceW: 32, SourceH: 32, ScaleX: 1, ScaleY: 1},
				})
				if err == nil || !strings.Contains(err.Error(), "stride 8 anchor 0") {
					t.Fatalf("non-finite %s did not produce a scoped decode error: %v", role, err)
				}
			})
		}
	}
}

func yunetTestDescriptor(t *testing.T) json.RawMessage {
	t.Helper()
	strides := make([]map[string]any, 0, 3)
	for _, stride := range []int{8, 16, 32} {
		strides = append(strides, map[string]any{
			"stride": stride, "cls": fmt.Sprintf("cls_%d", stride), "obj": fmt.Sprintf("obj_%d", stride),
			"bbox": fmt.Sprintf("bbox_%d", stride), "kps": fmt.Sprintf("kps_%d", stride),
		})
	}
	raw, err := json.Marshal(map[string]any{
		"kind": "detection.yunet", "threshold": 0.5,
		"nms": map[string]any{"iou": 0.3, "top_k": 300}, "strides": strides,
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func yunetTestResponse(input int, candidates []yunetTestCandidate) *infer.Response {
	response := &infer.Response{Outputs: map[string]*infer.TensorOutput{}}
	for _, stride := range []int{8, 16, 32} {
		anchors := (input / stride) * (input / stride)
		for _, role := range []struct {
			name     string
			channels int
			fill     float32
		}{{"cls", 1, -20}, {"obj", 1, -20}, {"bbox", 4, 0}, {"kps", 10, 0}} {
			name := fmt.Sprintf("%s_%d", role.name, stride)
			data := make([]float32, anchors*role.channels)
			if role.fill != 0 {
				for i := range data {
					data[i] = role.fill
				}
			}
			response.Outputs[name] = &infer.TensorOutput{Name: name, Shape: []int64{1, int64(anchors), int64(role.channels)}, Float32: data}
		}
	}
	for _, candidate := range candidates {
		cols := input / candidate.stride
		x, y := candidate.anchor%cols, candidate.anchor/cols
		cls := response.Outputs[fmt.Sprintf("cls_%d", candidate.stride)]
		obj := response.Outputs[fmt.Sprintf("obj_%d", candidate.stride)]
		bbox := response.Outputs[fmt.Sprintf("bbox_%d", candidate.stride)]
		kps := response.Outputs[fmt.Sprintf("kps_%d", candidate.stride)]
		cls.Float32[candidate.anchor] = candidate.cls
		obj.Float32[candidate.anchor] = candidate.obj
		base := candidate.anchor * 4
		bbox.Float32[base] = candidate.cx/float32(candidate.stride) - float32(x)
		bbox.Float32[base+1] = candidate.cy/float32(candidate.stride) - float32(y)
		bbox.Float32[base+2] = float32(math.Log(float64(candidate.w / float32(candidate.stride))))
		bbox.Float32[base+3] = float32(math.Log(float64(candidate.h / float32(candidate.stride))))
		for k, point := range candidate.points {
			kps.Float32[candidate.anchor*10+2*k] = point[0]/float32(candidate.stride) - float32(x)
			kps.Float32[candidate.anchor*10+2*k+1] = point[1]/float32(candidate.stride) - float32(y)
		}
	}
	return response
}
