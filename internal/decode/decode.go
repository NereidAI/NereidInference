// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package decode converts raw model-server tensors into normalized, bounded
// structures according to the model package descriptor. Decoders are keyed
// by a tensor-shape "kind" (never by a task, vendor or model name): adding a
// new head shape is a one-file registration, not a device-service fork.
package decode

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"sort"

	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

// inferTensor is a short alias for tensor outputs.
type inferTensor = infer.TensorOutput

// Reading is one normalized EdgeX reading emitted by a result.
type Reading struct {
	Name string
	// Type is one of "string", "int32", "int64", "float64", "bool".
	Type  string
	Value any
}

// Keypoint is one detected landmark in normalized source-frame coordinates.
type Keypoint struct {
	X     float32 `json:"x"`
	Y     float32 `json:"y"`
	Score float32 `json:"score"`
}

// Detection is a normalized object detection result.
type Detection struct {
	Class int     `json:"cls"`
	Label string  `json:"label"`
	Score float32 `json:"score"`
	// Box is normalized [x1,y1,x2,y2] in source-frame coordinates.
	Box       [4]float32 `json:"box"`
	Keypoints []Keypoint `json:"keypoints,omitempty"`
	// alignmentKeypoints preserves unclipped source geometry for consumers
	// whose transform would otherwise become degenerate at an image border.
	alignmentKeypoints []Keypoint
	// Mask is a normalized polygon (x,y pairs) for instance segmentation.
	Mask [][2]float32 `json:"mask,omitempty"`
}

// KeypointsForAlignment returns unclipped landmark geometry when the decoder
// retained it, falling back to the ordinary clipped result for other kinds.
func (d Detection) KeypointsForAlignment() []Keypoint {
	if len(d.alignmentKeypoints) == len(d.Keypoints) && len(d.alignmentKeypoints) > 0 {
		return d.alignmentKeypoints
	}
	return d.Keypoints
}

// Result is the output of a Decoder. Readings become EdgeX events; Draw
// renders the local MJPEG preview; JSON is the bounded structured payload.
type Result interface {
	Family() string
	JSON() any
	Readings(capBytes int) ([]Reading, error)
	// Draw overlays the result on an RGBA source-frame canvas. Text-only
	// results draw a caption line; no-op is allowed.
	Draw(canvas *image.RGBA, labels []string)
}

// Input bundles everything a decoder needs.
type Input struct {
	Manifest *model.Manifest
	Labels   []string
	Response *infer.Response
	Meta     preprocess.Meta
	// ConfOverride is the per-job threshold override (0 = descriptor value).
	ConfOverride float32
}

// Threshold returns the descriptor threshold unless the job overrode it.
func (in *Input) Threshold(descriptor float32) float32 {
	if in.ConfOverride > 0 {
		return in.ConfOverride
	}
	return descriptor
}

// Decoder decodes one tensor-layout family.
type Decoder interface {
	Kind() string
	Decode(in *Input) (Result, error)
}

// Factory builds a decoder from the raw descriptor JSON.
type Factory func(raw json.RawMessage) (Decoder, error)

// Registry maps decoder kinds to factories.
type Registry struct {
	factories map[string]Factory
}

// defaultFactories is populated once and shared by Registry and by the
// composite decoder, which instantiates named member codecs itself.
var defaultFactories = map[string]Factory{}

func registerFactory(f Factory) {
	d, err := f(nil)
	if err != nil {
		panic(err)
	}
	defaultFactories[d.Kind()] = f
}

// NewRegistry returns a registry preloaded with the built-in codec kinds.
func NewRegistry() *Registry {
	if len(defaultFactories) == 0 {
		for _, f := range []Factory{
			newAnchorsDecoder, newRowsDecoder, newYuNetDecoder,
			newProbsDecoder, newMultilabelDecoder, newRegressionDecoder,
			newSemanticSegDecoder, newInstanceSegDecoder,
			newKeypointsDecoder, newEmbeddingDecoder, newDenseMapDecoder,
			newImageTensorDecoder, newCTCDecoder, newCompositeDecoder,
			newPassthroughDecoder,
		} {
			registerFactory(f)
		}
	}
	r := &Registry{factories: map[string]Factory{}}
	for k, v := range defaultFactories {
		r.factories[k] = v
	}
	return r
}

// Build constructs the decoder for a manifest.
func (r *Registry) Build(m *model.Manifest) (Decoder, error) {
	kind, err := model.DecoderKind(m.Decoder)
	if err != nil {
		return nil, err
	}
	f, ok := r.factories[kind]
	if !ok {
		return nil, fmt.Errorf("decoder kind %q is not registered", kind)
	}
	return f(m.Decoder)
}

// nmsSpec is the shared non-maximum suppression configuration.
type nmsSpec struct {
	IoU      float32 `json:"iou"`
	Agnostic bool    `json:"agnostic"`
	TopK     int     `json:"top_k"`
}

func (n nmsSpec) orDefault() nmsSpec {
	if n.IoU <= 0 {
		n.IoU = 0.45
	}
	if n.TopK <= 0 {
		n.TopK = 300
	}
	return n
}

// outputTensor finds a tensor by descriptor output name or role fallback.
func shapeProduct(shape []int64) (int, bool) {
	var n int64 = 1
	for _, dim := range shape {
		if dim <= 0 {
			return 0, false
		}
		if n > math.MaxInt64/dim {
			return 0, false
		}
		n *= dim
	}
	if n > math.MaxInt {
		return 0, false
	}
	return int(n), true
}

func outputTensor(in *Input, name string) (*infer.TensorOutput, error) {
	if name == "" {
		for _, t := range in.Response.Outputs {
			return t, nil
		}
		return nil, fmt.Errorf("inference response has no outputs")
	}
	if t, ok := in.Response.Outputs[name]; ok {
		return t, nil
	}
	for _, spec := range in.Manifest.IO.Outputs {
		if spec.Role == name {
			if t, ok := in.Response.Outputs[spec.Name]; ok {
				return t, nil
			}
		}
	}
	return nil, fmt.Errorf("output tensor %q not found in response", name)
}

// clamp forces a value into the 0..1 range.
func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func safeInt32(n int) int32 {
	if n > math.MaxInt32 {
		return math.MaxInt32
	}
	if n < 0 {
		return 0
	}
	return int32(n)
}

// sigmoid computes the logistic function with overflow protection.
func sigmoid(x float32) float32 {
	if x < -80 {
		return 0
	}
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// topKVals returns indices of the k largest values, sorted descending.
func topKVals(vals []float32, k int) []int {
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return vals[idx[a]] > vals[idx[b]] })
	if k > 0 && len(idx) > k {
		idx = idx[:k]
	}
	return idx
}

func marshalBounded(v any, capBytes int) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	if capBytes > 0 && len(b) > capBytes {
		return "", fmt.Errorf("structured payload %d bytes exceeds cap %d", len(b), capBytes)
	}
	return string(b), nil
}
