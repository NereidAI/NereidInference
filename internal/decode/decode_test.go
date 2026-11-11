// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

func testInput(output *infer.TensorOutput, decoderJSON string) *Input {
	raw := json.RawMessage(decoderJSON)
	return &Input{
		Manifest: &model.Manifest{
			Model:   model.ModelSection{ID: "test-model", Version: "1.0.0"},
			Decoder: raw,
		},
		Labels: []string{"a", "b"},
		Response: &infer.Response{Outputs: map[string]*infer.TensorOutput{
			"output": output,
		}},
		Meta: preprocess.Meta{InputW: 100, InputH: 100, SourceW: 100, SourceH: 100, Scale: 1},
	}
}

func TestAnchorsDecoder(t *testing.T) {
	// Layout [1, 4+classes, anchors]: two classes, one strong candidate.
	data := make([]float32, 6*3)
	set := func(channel, anchor int, v float32) { data[channel*3+anchor] = v }
	set(0, 0, 50) // cx
	set(1, 0, 50) // cy
	set(2, 0, 20) // w
	set(3, 0, 20) // h
	set(5, 0, 0.9)
	output := &infer.TensorOutput{Name: "output", DataType: "FP32", Shape: []int64{1, 6, 3}, Float32: data}

	d, err := newAnchorsDecoder(json.RawMessage(`{"kind":"detection.anchors","output":"output","threshold":0.25}`))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{"kind":"detection.anchors"}`))
	if err != nil {
		t.Fatal(err)
	}
	det := result.(*DetectionResult)
	if len(det.Detections) != 1 {
		t.Fatalf("expected one detection, got %d", len(det.Detections))
	}
	got := det.Detections[0]
	if got.Label != "b" || got.Class != 1 || got.Score < 0.89 {
		t.Fatalf("unexpected detection: %+v", got)
	}
	wantBox := [4]float32{0.4, 0.4, 0.6, 0.6}
	if got.Box != wantBox {
		t.Fatalf("box = %v, want %v", got.Box, wantBox)
	}
	readings, err := result.Readings(1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(readings) < 4 {
		t.Fatalf("expected count, JSON and top readings, got %d", len(readings))
	}
}

func TestProbsDecoder(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "output", DataType: "FP32", Shape: []int64{1, 3},
		Float32: []float32{2, 1, 0},
	}
	d, err := newProbsDecoder(json.RawMessage(`{"kind":"classification.probs","output":"output","softmax":true,"top_k":2}`))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{}`))
	if err != nil {
		t.Fatal(err)
	}
	caption := result.(*CaptionResult)
	if len(caption.Rank) != 2 || caption.Rank[0].Label != "a" {
		t.Fatalf("unexpected ranking: %+v", caption.Rank)
	}
	if caption.Rank[0].Score <= caption.Rank[1].Score || caption.Rank[0].Score < 0.6 {
		t.Fatalf("probabilities not softmax-ordered: %+v", caption.Rank)
	}
}

func TestMultilabelDecoder(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "output", DataType: "FP32", Shape: []int64{2},
		Float32: []float32{10, -10},
	}
	d, err := newMultilabelDecoder(json.RawMessage(`{"kind":"classification.multilabel","output":"output","threshold":0.5}`))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{}`))
	if err != nil {
		t.Fatal(err)
	}
	caption := result.(*CaptionResult)
	if len(caption.Rank) != 1 || caption.Rank[0].Label != "a" {
		t.Fatalf("unexpected attributes: %+v", caption.Rank)
	}
}

func TestRegressionDecoder(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "output", DataType: "FP32", Shape: []int64{1, 2},
		Float32: []float32{31.5, 1.25},
	}
	spec := `{"kind":"regression.values","output":"output","names":[{"channel":0,"name":"Temperature"},{"channel":1,"name":"Scale"}]}`
	d, err := newRegressionDecoder(json.RawMessage(spec))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{}`))
	if err != nil {
		t.Fatal(err)
	}
	scalar := result.(*ScalarResult)
	if scalar.Values["Temperature"] != 31.5 || scalar.Values["Scale"] != 1.25 {
		t.Fatalf("unexpected values: %#v", scalar.Values)
	}
	readings, err := result.Readings(1 << 20)
	if err != nil || len(readings) < 3 {
		t.Fatalf("readings=%v err=%v", readings, err)
	}
}

func TestStretchBoxUsesIndependentAxisScales(t *testing.T) {
	meta := preprocess.Meta{
		InputW: 4, InputH: 4, SourceW: 2, SourceH: 1,
		ScaleX: 2, ScaleY: 4,
	}
	box := normalizeAndUnmap(meta, 0, 0, 2, 2, false)
	want := [4]float32{0, 0, 0.5, 0.5}
	if box != want {
		t.Fatalf("stretch box = %v, want %v", box, want)
	}
}

func TestSemanticSegmentationUnmapsLetterboxBox(t *testing.T) {
	// Two classes over a 4x2 mask. Class 1 occupies only the two central
	// rows where the source image sits after top/bottom letterbox padding.
	data := make([]float32, 2*4*2)
	for i := 0; i < 4*2; i++ {
		data[i] = -10 // class 0 background/padding
	}
	for y := 1; y <= 2; y++ {
		for x := 0; x < 2; x++ {
			data[4*2+y*2+x] = 10
		}
	}
	output := &infer.TensorOutput{Name: "seg", DataType: "FP32", Shape: []int64{1, 2, 4, 2}, Float32: data}
	raw := json.RawMessage(`{"kind":"segmentation.semantic","output":"seg"}`)
	d, err := newSemanticSegDecoder(raw)
	if err != nil {
		t.Fatal(err)
	}
	in := testInput(output, `{}`)
	in.Response.Outputs["seg"] = output
	in.Meta = preprocess.Meta{InputW: 4, InputH: 4, SourceW: 2, SourceH: 1, Scale: 2, PadY: 1}
	result, err := d.Decode(in)
	if err != nil {
		t.Fatal(err)
	}
	seg := result.(*SegmentationResult)
	if len(seg.Regions) != 1 {
		t.Fatalf("regions = %+v", seg.Regions)
	}
	box := seg.Regions[0].Box
	want := [4]float32{0, 0, 1, 1}
	if box != want {
		t.Fatalf("unmapped box = %v, want %v", box, want)
	}
}

func TestCTCRawLogitsArgmax(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "output", DataType: "FP32", Shape: []int64{1, 2},
		Float32: []float32{-5, -2},
	}
	d, err := newCTCDecoder(json.RawMessage(`{"output":"output","activation":"none"}`))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{}`))
	if err != nil {
		t.Fatal(err)
	}
	text := result.(*TextResult).Text
	if text != "b" {
		t.Fatalf("CTC text = %q, want b", text)
	}
}

func TestDecoderMalformedShapesReturnErrors(t *testing.T) {
	cases := []struct {
		name string
		out  *infer.TensorOutput
		raw  string
	}{
		{"dense", &infer.TensorOutput{Name: "o", Shape: []int64{1, 1, 0, 1}}, `{"kind":"dense.map","output":"o"}`},
		{"image", &infer.TensorOutput{Name: "o", Shape: []int64{1, 3, 0, 1}}, `{"kind":"image.tensor","output":"o"}`},
		{"embedding", &infer.TensorOutput{Name: "o", Shape: []int64{2, 2}, Float32: []float32{1, 2, 3, 4}}, `{"kind":"embeddings.vector","output":"o"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRegistry()
			m := &model.Manifest{Decoder: json.RawMessage(tc.raw)}
			d, err := r.Build(m)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("decoder panicked: %v", recovered)
				}
			}()
			in := testInput(tc.out, `{}`)
			in.Response.Outputs[tc.out.Name] = tc.out
			if _, err := d.Decode(in); err == nil {
				t.Fatal("expected malformed shape error")
			}
		})
	}
}

func TestRegressionIgnoresNegativeChannels(t *testing.T) {
	output := &infer.TensorOutput{Shape: []int64{1}, Float32: []float32{1.5}}
	d, err := newRegressionDecoder(json.RawMessage(`{"names":[{"channel":-1,"name":"Bad"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	result, err := d.Decode(testInput(output, `{}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.(*ScalarResult).Values) != 0 {
		t.Fatalf("negative channel was not ignored: %#v", result.(*ScalarResult).Values)
	}
}

func TestKeypointsRejectsStrideBelowThree(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "o", Shape: []int64{1, 4, 1},
		// count=2, stride=2 nominally fits channels, but each keypoint reads
		// three consecutive channels and must be rejected before indexing.
		Float32: []float32{0, 0, 1, 0},
	}
	d, err := newKeypointsDecoder(json.RawMessage(`{"output":"o","start":0,"count":2,"stride":2}`))
	if err != nil {
		t.Fatal(err)
	}
	in := testInput(output, `{}`)
	in.Response.Outputs["o"] = output
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("keypoints decoder panicked: %v", recovered)
		}
	}()
	if _, err := d.Decode(in); err == nil {
		t.Fatal("expected stride below three to be rejected")
	}
}

func TestRowsDecoderGuardsPartialFinalRow(t *testing.T) {
	output := &infer.TensorOutput{
		Name: "o", Shape: []int64{1, 2, 7},
		// Only a partial second row is present; it must be skipped rather than indexed.
		Float32: []float32{1, 1, .9, .1, .2, .3, .4, 1, 1, .9, .1, .2},
	}
	raw := `{"output":"o","fields":{"label":1,"conf":2,"y2":3,"x1":4,"y1":5,"x2":6},"normalized_coords":true}`
	d, err := newRowsDecoder(json.RawMessage(raw))
	if err != nil {
		t.Fatal(err)
	}
	in := testInput(output, `{}`)
	in.Response.Outputs["o"] = output
	result, err := d.Decode(in)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(result.(*DetectionResult).Detections); got != 1 {
		t.Fatalf("detections = %d, want 1", got)
	}
}

func TestRegistryRejectsUnknownKind(t *testing.T) {
	r := NewRegistry()
	m := &model.Manifest{Decoder: json.RawMessage(`{"kind":"vendor.special"}`)}
	if _, err := r.Build(m); err == nil {
		t.Fatal("expected unknown decoder kind to fail")
	}
}
