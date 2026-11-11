// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// yunetStrideSpec explicitly binds one stride's four output tensors. The kind
// describes a tensor-layout family only; no model id, runtime name or filename
// participates in decoder selection or execution.
type yunetStrideSpec struct {
	Stride int    `json:"stride"`
	Cls    string `json:"cls"`
	Obj    string `json:"obj"`
	BBox   string `json:"bbox"`
	KPS    string `json:"kps"`
}

type yunetSpec struct {
	Threshold  float32           `json:"threshold"`
	Activation string            `json:"activation"`
	NMS        nmsSpec           `json:"nms"`
	Strides    []yunetStrideSpec `json:"strides"`
}

type yunetDecoder struct{ spec yunetSpec }

func newYuNetDecoder(raw json.RawMessage) (Decoder, error) {
	d := &yunetDecoder{spec: yunetSpec{Threshold: 0.6, Activation: "probability", NMS: nmsSpec{IoU: 0.3, TopK: 300}}}
	if len(raw) == 0 {
		// Registry probing calls every factory with nil to discover Kind. Real
		// manifests always supply a decoder object and take the strict path.
		return d, nil
	}
	if err := json.Unmarshal(raw, &d.spec); err != nil {
		return nil, err
	}
	if d.spec.Threshold <= 0 || d.spec.Threshold > 1 {
		return nil, fmt.Errorf("detection.yunet threshold must be in (0,1], got %g", d.spec.Threshold)
	}
	if d.spec.Activation != "probability" && d.spec.Activation != "logit" {
		return nil, fmt.Errorf("detection.yunet activation must be probability or logit, got %q", d.spec.Activation)
	}
	if len(d.spec.Strides) != 3 {
		return nil, fmt.Errorf("detection.yunet requires cls/obj/bbox/kps outputs for strides 8, 16 and 32")
	}
	seen := map[int]bool{}
	for _, stride := range d.spec.Strides {
		if stride.Stride != 8 && stride.Stride != 16 && stride.Stride != 32 {
			return nil, fmt.Errorf("detection.yunet has unsupported stride %d", stride.Stride)
		}
		if seen[stride.Stride] {
			return nil, fmt.Errorf("detection.yunet has duplicate stride %d", stride.Stride)
		}
		seen[stride.Stride] = true
		if stride.Cls == "" || stride.Obj == "" || stride.BBox == "" || stride.KPS == "" {
			return nil, fmt.Errorf("detection.yunet stride %d must explicitly name cls, obj, bbox and kps outputs", stride.Stride)
		}
	}
	sort.Slice(d.spec.Strides, func(i, j int) bool { return d.spec.Strides[i].Stride < d.spec.Strides[j].Stride })
	return d, nil
}

func (d *yunetDecoder) Kind() string { return "detection.yunet" }

// Decode consumes YuNet-style distance-to-grid boxes and five landmark pairs.
// Every candidate owns its landmarks before NMS, so suppression cannot detach
// a high-scoring box from the wrong face geometry.
func (d *yunetDecoder) Decode(in *Input) (Result, error) {
	if in.Meta.InputW <= 0 || in.Meta.InputH <= 0 || in.Meta.SourceW <= 0 || in.Meta.SourceH <= 0 {
		return nil, fmt.Errorf("detection.yunet requires positive preprocess/source dimensions")
	}
	threshold := in.Threshold(d.spec.Threshold)
	candidates := make([]Detection, 0)
	for _, strideSpec := range d.spec.Strides {
		stride := strideSpec.Stride
		if in.Meta.InputW%stride != 0 || in.Meta.InputH%stride != 0 {
			return nil, fmt.Errorf("detection.yunet input %dx%d is not divisible by stride %d", in.Meta.InputW, in.Meta.InputH, stride)
		}
		cols, rows := in.Meta.InputW/stride, in.Meta.InputH/stride
		cls, anchors, err := yunetOutput(in, strideSpec.Cls, 1, rows, cols)
		if err != nil {
			return nil, err
		}
		obj, objAnchors, err := yunetOutput(in, strideSpec.Obj, 1, rows, cols)
		if err != nil {
			return nil, err
		}
		bbox, bboxAnchors, err := yunetOutput(in, strideSpec.BBox, 4, rows, cols)
		if err != nil {
			return nil, err
		}
		kps, kpsAnchors, err := yunetOutput(in, strideSpec.KPS, 10, rows, cols)
		if err != nil {
			return nil, err
		}
		if anchors != rows*cols || objAnchors != anchors || bboxAnchors != anchors || kpsAnchors != anchors {
			return nil, fmt.Errorf("detection.yunet stride %d output anchors mismatch: cls=%d obj=%d bbox=%d kps=%d, want %d", stride, anchors, objAnchors, bboxAnchors, kpsAnchors, rows*cols)
		}
		for a := 0; a < anchors; a++ {
			clsValue, objValue := cls[a], obj[a]
			if !finiteBox(clsValue, objValue) {
				return nil, fmt.Errorf("detection.yunet stride %d anchor %d has non-finite cls/obj values", stride, a)
			}
			clsScore, objScore := clsValue, objValue
			if d.spec.Activation == "logit" {
				clsScore, objScore = sigmoid(clsScore), sigmoid(objScore)
			} else {
				// OpenCV YuNet exports probabilities, not logits. Match its
				// clamp-then-sqrt(cls*obj) semantics; applying another sigmoid
				// would inflate low-confidence candidates into false positives.
				clsScore, objScore = clamp(clsScore), clamp(objScore)
			}
			score := float32(math.Sqrt(float64(clsScore * objScore)))
			if !finiteBox(score) {
				return nil, fmt.Errorf("detection.yunet stride %d anchor %d has a non-finite score", stride, a)
			}
			if score < threshold {
				continue
			}
			x, y := a%cols, a/cols
			bboxBase, kpsBase := a*4, a*10
			dx, dy, logW, logH := bbox[bboxBase], bbox[bboxBase+1], bbox[bboxBase+2], bbox[bboxBase+3]
			if !finiteBox(dx, dy, logW, logH) {
				return nil, fmt.Errorf("detection.yunet stride %d anchor %d has non-finite bbox values", stride, a)
			}
			w := float32(math.Exp(float64(logW))) * float32(stride)
			h := float32(math.Exp(float64(logH))) * float32(stride)
			cx := (dx + float32(x)) * float32(stride)
			cy := (dy + float32(y)) * float32(stride)
			if !finiteBox(cx, cy, w, h) || w <= 0 || h <= 0 {
				return nil, fmt.Errorf("detection.yunet stride %d anchor %d has invalid decoded box geometry", stride, a)
			}
			det := Detection{
				Class: 0, Label: "face", Score: score,
				// NMS runs in input-pixel geometry. Applying independent source
				// axis scales before IoU would change suppression under stretch.
				Box:       [4]float32{cx - w/2, cy - h/2, cx + w/2, cy + h/2},
				Keypoints: make([]Keypoint, 0, 5),
			}
			for k := 0; k < 5; k++ {
				kx := (kps[kpsBase+2*k] + float32(x)) * float32(stride)
				ky := (kps[kpsBase+2*k+1] + float32(y)) * float32(stride)
				if !finiteBox(kx, ky) {
					return nil, fmt.Errorf("detection.yunet stride %d anchor %d has non-finite landmark geometry", stride, a)
				}
				det.Keypoints = append(det.Keypoints, Keypoint{X: kx, Y: ky, Score: score})
			}
			candidates = append(candidates, det)
		}
	}
	kept := nonMaxSuppression(candidates, d.spec.NMS.orDefault())
	mapped := make([]Detection, 0, len(kept))
	for _, det := range kept {
		rawBox, rawKeypoints := det.Box, det.Keypoints
		det.Box = normalizeAndUnmap(in.Meta, rawBox[0], rawBox[1], rawBox[2], rawBox[3], false)
		// Fully out-of-frame candidates can collapse during clipping. They are
		// not valid ROIs and must not poison other healthy faces in the frame.
		if det.Box[2] <= det.Box[0] || det.Box[3] <= det.Box[1] {
			continue
		}
		det.alignmentKeypoints = append([]Keypoint(nil), rawKeypoints...)
		for i, keypoint := range rawKeypoints {
			x := keypoint.X / float32(in.Meta.InputW)
			y := keypoint.Y / float32(in.Meta.InputH)
			u, v, _, _ := in.Meta.UnmapBox(x, y, x, y)
			det.Keypoints[i].X, det.Keypoints[i].Y = u, v
			u, v = in.Meta.UnmapPoint(x, y)
			det.alignmentKeypoints[i].X, det.alignmentKeypoints[i].Y = u, v
		}
		mapped = append(mapped, det)
	}
	return newDetectionResult(in, d.Kind(), mapped), nil
}

// yunetOutput reads the contiguous anchor-major layouts emitted by common
// YuNet exports: [A,C], [1,A,C] and NHWC [1,H,W,C].
func yunetOutput(in *Input, name string, channels, rows, cols int) ([]float32, int, error) {
	t, err := outputTensor(in, name)
	if err != nil {
		return nil, 0, err
	}
	anchors := 0
	switch len(t.Shape) {
	case 2:
		if t.Shape[1] != int64(channels) {
			return nil, 0, badShape("detection.yunet "+name, fmt.Sprintf("[A,%d]", channels), t.Shape)
		}
		anchors = int(t.Shape[0])
	case 3:
		if t.Shape[0] != 1 || t.Shape[2] != int64(channels) {
			return nil, 0, badShape("detection.yunet "+name, fmt.Sprintf("[1,A,%d]", channels), t.Shape)
		}
		anchors = int(t.Shape[1])
	case 4:
		if t.Shape[0] != 1 || t.Shape[1] != int64(rows) || t.Shape[2] != int64(cols) || t.Shape[3] != int64(channels) {
			return nil, 0, badShape("detection.yunet "+name, fmt.Sprintf("[1,%d,%d,%d]", rows, cols, channels), t.Shape)
		}
		anchors = rows * cols
	default:
		return nil, 0, badShape("detection.yunet "+name, "anchor-major tensor", t.Shape)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || anchors <= 0 || required != anchors*channels || len(t.Float32) != required {
		return nil, 0, badShape("detection.yunet "+name, "complete, positive anchor-major tensor", t.Shape)
	}
	return t.Float32, anchors, nil
}
