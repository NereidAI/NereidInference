// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"strconv"

	"github.com/edgexfoundry/device-inference/internal/infer"
)

// RankEntry is one class with its probability.
type RankEntry struct {
	Class int     `json:"cls"`
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// CaptionResult covers single-label and multi-label classification.
type CaptionResult struct {
	Schema   string      `json:"schema"`
	ModelRef modelRef    `json:"model"`
	Frame    frameRef    `json:"frame"`
	Rank     []RankEntry `json:"rank"`

	family   string
	jsonName string
}

// Family returns the result family.
func (r *CaptionResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *CaptionResult) JSON() any { return r }

// Readings emits the top label/score and a bounded top-k JSON point.
func (r *CaptionResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	out := []Reading{{Name: r.jsonName, Type: "string", Value: payload}}
	if len(r.Rank) > 0 {
		out = append(out,
			Reading{Name: "Top1Label", Type: "string", Value: r.Rank[0].Label},
			Reading{Name: "Top1Score", Type: "float64", Value: float64(r.Rank[0].Score)},
		)
	}
	return out, nil
}

// Draw paints the top label as a caption.
func (r *CaptionResult) Draw(canvas *image.RGBA, _ []string) {
	if len(r.Rank) > 0 {
		caption(canvas, 4, 14, r.Rank[0].Label+" "+strconv.FormatFloat(float64(r.Rank[0].Score), 'f', 2, 32))
	}
}

// ScalarResult is a descriptor-named set of scalar readings (regression).
// EdgeX profiles need stable resource names, so every result additionally
// exposes RegressionValue for the first mapped channel and RegressionJSON
// with the complete named map.
type ScalarResult struct {
	Schema   string             `json:"schema"`
	ModelRef modelRef           `json:"model"`
	Values   map[string]float64 `json:"values"`

	family string
	order  []string
}

// Family returns the result family.
func (r *ScalarResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *ScalarResult) JSON() any { return r }

// Readings returns the complete map as JSON plus a stable scalar point.
func (r *ScalarResult) Readings(capBytes int) ([]Reading, error) {
	out := []Reading{{Name: "RegressionValue", Type: "float64", Value: float64(0)}}
	if len(r.order) > 0 {
		out[0].Value = r.Values[r.order[0]]
	}
	if payload, err := marshalBounded(r, capBytes); err == nil {
		out = append(out, Reading{Name: "RegressionJSON", Type: "string", Value: payload})
	}
	for _, name := range r.order {
		out = append(out, Reading{Name: name, Type: "float64", Value: r.Values[name]})
	}
	return out, nil
}

// Draw is a no-op for scalar results.
func (r *ScalarResult) Draw(*image.RGBA, []string) {}

type probsSpec struct {
	Output    string  `json:"output"`
	Softmax   bool    `json:"softmax"`
	TopK      int     `json:"top_k"`
	Threshold float32 `json:"threshold"`
}

type probsDecoder struct{ spec probsSpec }

func newProbsDecoder(raw json.RawMessage) (Decoder, error) {
	d := &probsDecoder{spec: probsSpec{Softmax: true, TopK: 5, Threshold: 0.01}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *probsDecoder) Kind() string { return "classification.probs" }

// Decode handles [1,nc] or [nc] probability tensors.
func (d *probsDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	vals, err := lastDim(t)
	if err != nil {
		return nil, err
	}
	if d.spec.Softmax {
		vals = softmaxCopy(vals)
	}
	k := d.spec.TopK
	if k <= 0 {
		k = 5
	}
	return &CaptionResult{
		Schema:   "nereid.classification/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:    frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		Rank:     ranking(vals, in.Labels, k, d.spec.Threshold),
		family:   "classification",
		jsonName: "ProbsJSON",
	}, nil
}

type multilabelSpec struct {
	Output     string  `json:"output"`
	Threshold  float32 `json:"threshold"`
	TopK       int     `json:"top_k"`
	Activation string  `json:"activation"` // sigmoid (default) or none
}

type multilabelDecoder struct{ spec multilabelSpec }

func newMultilabelDecoder(raw json.RawMessage) (Decoder, error) {
	d := &multilabelDecoder{spec: multilabelSpec{Threshold: 0.5, TopK: 20}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *multilabelDecoder) Kind() string { return "classification.multilabel" }

// Decode applies an independent sigmoid threshold per class (PPE/attributes).
func (d *multilabelDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	vals, err := lastDim(t)
	if err != nil {
		return nil, err
	}
	switch d.spec.Activation {
	case "", "sigmoid":
		for i := range vals {
			vals[i] = sigmoid(vals[i])
		}
	case "none":
	default:
		return nil, fmt.Errorf("classification.multilabel activation must be sigmoid or none, got %q", d.spec.Activation)
	}
	return &CaptionResult{
		Schema:   "nereid.multilabel/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:    frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		Rank:     ranking(vals, in.Labels, d.spec.TopK, d.spec.Threshold),
		family:   "classification",
		jsonName: "AttributesJSON",
	}, nil
}

type regNameEntry struct {
	Channel int    `json:"channel"`
	Name    string `json:"name"`
}

type regressionSpec struct {
	Output string         `json:"output"`
	Names  []regNameEntry `json:"names"`
}

type regressionDecoder struct{ spec regressionSpec }

func newRegressionDecoder(raw json.RawMessage) (Decoder, error) {
	d := &regressionDecoder{}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *regressionDecoder) Kind() string { return "regression.values" }

// Decode emits one float64 reading per mapped channel.
func (d *regressionDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	vals, err := lastDim(t)
	if err != nil {
		return nil, err
	}
	values := map[string]float64{}
	names := make([]string, 0, len(d.spec.Names))
	for _, n := range d.spec.Names {
		if n.Channel < 0 || n.Channel >= len(vals) || n.Name == "" {
			continue
		}
		values[n.Name] = float64(vals[n.Channel])
		names = append(names, n.Name)
	}
	return &ScalarResult{
		Schema:   "nereid.regression/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Values:   values,
		family:   "regression",
		order:    names,
	}, nil
}

func ranking(vals []float32, labels []string, topK int, threshold float32) []RankEntry {
	idx := topKVals(vals, topK)
	out := make([]RankEntry, 0, len(idx))
	for _, i := range idx {
		if vals[i] < threshold {
			continue
		}
		out = append(out, RankEntry{Class: i, Label: labelFor(labels, i), Score: vals[i]})
	}
	return out
}

func lastDim(t *infer.TensorOutput) ([]float32, error) {
	if t == nil || len(t.Shape) == 0 {
		return nil, fmt.Errorf("tensor has no shape")
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || required > len(t.Float32) {
		return nil, fmt.Errorf("tensor with shape %v is truncated", t.Shape)
	}
	n := int(t.Shape[len(t.Shape)-1])
	if n <= 0 {
		return nil, fmt.Errorf("tensor last dimension must be positive, got %d", n)
	}
	return t.Float32[len(t.Float32)-n:], nil
}

func softmaxCopy(v []float32) []float32 {
	maxV := float32(-math.MaxFloat32)
	for _, x := range v {
		if x > maxV {
			maxV = x
		}
	}
	sum := float32(0)
	out := make([]float32, len(v))
	for i, x := range v {
		e := float32(0)
		if x-maxV > -80 {
			e = float32(math.Exp(float64(x - maxV)))
		}
		out[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range out {
			out[i] /= sum
		}
	}
	return out
}
