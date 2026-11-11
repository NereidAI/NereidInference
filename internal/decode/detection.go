// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"image"
	"math"

	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

// DetectionResult is the common result of detection codecs.
type DetectionResult struct {
	Schema     string      `json:"schema"`
	ModelRef   modelRef    `json:"model"`
	Frame      frameRef    `json:"frame"`
	Detections []Detection `json:"detections"`

	kind   string
	labels []string
}

type modelRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

type frameRef struct {
	Width  int `json:"w"`
	Height int `json:"h"`
}

// Family returns the result family.
func (r *DetectionResult) Family() string { return "detection" }

// JSON returns the structured payload.
func (r *DetectionResult) JSON() any { return r }

// Readings emits the detection/v1 point table.
func (r *DetectionResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	out := []Reading{
		{Name: "DetectionCount", Type: "int32", Value: safeInt32(len(r.Detections))},
		{Name: "DetectionJSON", Type: "string", Value: payload},
	}
	if len(r.Detections) > 0 {
		top := r.Detections[0]
		out = append(out,
			Reading{Name: "Top1Label", Type: "string", Value: top.Label},
			Reading{Name: "Top1Score", Type: "float64", Value: float64(top.Score)},
		)
	}
	return out, nil
}

// Draw renders boxes with labels on the preview canvas.
func (r *DetectionResult) Draw(canvas *image.RGBA, _ []string) {
	for _, d := range r.Detections {
		drawBox(canvas, d.Box, d.Class, d.Label, d.Score)
	}
}

func newDetectionResult(in *Input, kind string, ds []Detection) *DetectionResult {
	return &DetectionResult{
		Schema:     "nereid.detection/1",
		ModelRef:   modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:      frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		Detections: ds,
		kind:       kind,
		labels:     in.Labels,
	}
}

func labelFor(labels []string, id int) string {
	if id >= 0 && id < len(labels) && labels[id] != "" {
		return labels[id]
	}
	return fmt.Sprintf("#%d", id)
}

// ---- detection.anchors (YOLOv8/v11 style [1,4+nc,A]) ----

type anchorsSpec struct {
	Output    string  `json:"output"`
	Threshold float32 `json:"threshold"`
	NMS       nmsSpec `json:"nms"`
}

type anchorsDecoder struct{ spec anchorsSpec }

func newAnchorsDecoder(raw json.RawMessage) (Decoder, error) {
	d := &anchorsDecoder{spec: anchorsSpec{Threshold: 0.25, NMS: nmsSpec{IoU: 0.45, TopK: 300}}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *anchorsDecoder) Kind() string { return "detection.anchors" }

// Decode parses the canonical [1,4+nc,A] anchor tensor.
func (d *anchorsDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 3 || t.Shape[0] != 1 {
		return nil, fmt.Errorf("anchors decoder expects [1,C,A], got %v", t.Shape)
	}
	channels, anchors := int(t.Shape[1]), int(t.Shape[2])
	if channels <= 0 || anchors <= 0 {
		return nil, badShape("detection.anchors", "positive channels and anchors", t.Shape)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, fmt.Errorf("anchors tensor truncated or too large")
	}
	nc := channels - 4
	if nc <= 0 {
		return nil, fmt.Errorf("anchors channels %d leave no score classes", channels)
	}
	threshold := in.Threshold(d.spec.Threshold)
	var candidates []Detection
	for a := 0; a < anchors; a++ {
		cx, cy, w, h := t.Float32[0*anchors+a], t.Float32[1*anchors+a], t.Float32[2*anchors+a], t.Float32[3*anchors+a]
		bestClass, bestScore := -1, threshold
		for c := 0; c < nc; c++ {
			if s := t.Float32[(4+c)*anchors+a]; s > bestScore {
				bestScore, bestClass = s, c
			}
		}
		if bestClass < 0 || !finiteBox(cx, cy, w, h) {
			continue
		}
		box := normalizeAndUnmap(in.Meta, cx-w/2, cy-h/2, cx+w/2, cy+h/2, false)
		candidates = append(candidates, Detection{
			Class: bestClass, Label: labelFor(in.Labels, bestClass), Score: bestScore, Box: box,
		})
	}
	kept := nonMaxSuppression(candidates, d.spec.NMS.orDefault())
	return newDetectionResult(in, d.Kind(), kept), nil
}

// ---- detection.rows (per-row fixed-layout outputs) ----

type rowsSpec struct {
	Output    string  `json:"output"`
	Threshold float32 `json:"threshold"`
	NMS       nmsSpec `json:"nms"`
	// Fields gives column indices; defaults match the legacy OVMS example
	// [image_id, label, confidence, x1, y1, x2, y2].
	Fields struct {
		Label int `json:"label"`
		Conf  int `json:"conf"`
		X1    int `json:"x1"`
		Y1    int `json:"y1"`
		X2    int `json:"x2"`
		Y2    int `json:"y2"`
	} `json:"fields"`
	// NormalizedCoords is true when values are already in 0..1 input space.
	NormalizedCoords bool `json:"normalized_coords"`
}

type rowsDecoder struct{ spec rowsSpec }

func newRowsDecoder(raw json.RawMessage) (Decoder, error) {
	d := &rowsDecoder{}
	d.spec.Threshold = 0.5
	d.spec.NMS = nmsSpec{IoU: 0.45, TopK: 300}
	d.spec.Fields = struct {
		Label int `json:"label"`
		Conf  int `json:"conf"`
		X1    int `json:"x1"`
		Y1    int `json:"y1"`
		X2    int `json:"x2"`
		Y2    int `json:"y2"`
	}{Label: 1, Conf: 2, X1: 3, Y1: 4, X2: 5, Y2: 6}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *rowsDecoder) Kind() string { return "detection.rows" }

// Decode parses row-layout detections ([N,A,K] or [A,K]).
func (d *rowsDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	rows, stride, err := flattenRows(t)
	if err != nil {
		return nil, err
	}
	f := d.spec.Fields
	for _, idx := range []int{f.Label, f.Conf, f.X1, f.Y1, f.X2, f.Y2} {
		if idx < 0 || idx >= stride {
			return nil, fmt.Errorf("rows field index %d is outside stride %d", idx, stride)
		}
	}
	threshold := in.Threshold(d.spec.Threshold)
	var candidates []Detection
	for i := 0; i < rows; i++ {
		base := i * stride
		if base+stride > len(t.Float32) {
			break
		}
		score := t.Float32[base+f.Conf]
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score < threshold {
			continue
		}
		class := int(t.Float32[base+f.Label] + 0.5)
		if class < 0 {
			continue
		}
		x1, y1 := t.Float32[base+f.X1], t.Float32[base+f.Y1]
		x2, y2 := t.Float32[base+f.X2], t.Float32[base+f.Y2]
		if !finiteBox(x1, y1, x2, y2) {
			continue
		}
		box := normalizeAndUnmap(in.Meta, x1, y1, x2, y2, d.spec.NormalizedCoords)
		candidates = append(candidates, Detection{
			Class: class, Label: labelFor(in.Labels, class), Score: score, Box: box,
		})
	}
	kept := nonMaxSuppression(candidates, d.spec.NMS.orDefault())
	return newDetectionResult(in, d.Kind(), kept), nil
}

// flattenRows reduces [1,A,K] and [A,K] output shapes to (rows, stride).
func flattenRows(t *inferTensor) (int, int, error) {
	if t == nil {
		return 0, 0, fmt.Errorf("nil tensor")
	}
	switch len(t.Shape) {
	case 2:
		return int(t.Shape[0]), int(t.Shape[1]), nil
	case 3:
		if t.Shape[0] != 1 {
			return 0, 0, fmt.Errorf("rows decoder supports batch 1, got %v", t.Shape)
		}
		return int(t.Shape[1]), int(t.Shape[2]), nil
	default:
		return 0, 0, fmt.Errorf("rows decoder cannot read shape %v", t.Shape)
	}
}

// normalizeAndUnmap maps a box from input-pixel (or already-normalized)
// coordinates into normalized source-frame coordinates.
func finiteBox(values ...float32) bool {
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func normalizeAndUnmap(meta preprocess.Meta, x1, y1, x2, y2 float32, alreadyNormalized bool) [4]float32 {
	if !alreadyNormalized {
		x1 /= float32(meta.InputW)
		y1 /= float32(meta.InputH)
		x2 /= float32(meta.InputW)
		y2 /= float32(meta.InputH)
	}
	x1, y1, x2, y2 = meta.UnmapBox(clamp(x1), clamp(y1), clamp(x2), clamp(y2))
	return [4]float32{x1, y1, x2, y2}
}

// nonMaxSuppression filters candidates per class.
func nonMaxSuppression(ds []Detection, spec nmsSpec) []Detection {
	idx := nonMaxSuppressIndices(len(ds),
		func(i int) float32 { return ds[i].Score },
		func(i int) int { return ds[i].Class },
		func(i int) [4]float32 { return ds[i].Box },
		spec)
	kept := make([]Detection, 0, len(idx))
	for _, i := range idx {
		kept = append(kept, ds[i])
	}
	return kept
}

// nonMaxSuppressIndices returns the surviving candidate indices; candidates
// above TopK by score are suppressed first.
func nonMaxSuppressIndices(n int, score func(int) float32, class func(int) int, box func(int) [4]float32, spec nmsSpec) []int {
	scores := make([]float32, n)
	for i := 0; i < n; i++ {
		scores[i] = score(i)
	}
	order := topKVals(scores, spec.TopK)
	used := make([]bool, n)
	var kept []int
	for _, i := range order {
		if used[i] {
			continue
		}
		used[i] = true
		for _, j := range order {
			if used[j] || (!spec.Agnostic && class(i) != class(j)) {
				continue
			}
			if iou(box(i), box(j)) > spec.IoU {
				used[j] = true
			}
		}
		kept = append(kept, i)
	}
	return kept
}

func iou(a, b [4]float32) float32 {
	ix1, iy1 := maxF32(a[0], b[0]), maxF32(a[1], b[1])
	ix2, iy2 := minF32(a[2], b[2]), minF32(a[3], b[3])
	iw, ih := maxF32(0, ix2-ix1), maxF32(0, iy2-iy1)
	inter := iw * ih
	ua := (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
	if ua <= 0 {
		return 0
	}
	return inter / ua
}

func maxF32(a, b float32) float32 { return float32(math.Max(float64(a), float64(b))) }
func minF32(a, b float32) float32 { return float32(math.Min(float64(a), float64(b))) }
