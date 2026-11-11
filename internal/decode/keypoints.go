// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
)

// KeypointResult is the normalized pose/landmark output.
type KeypointResult struct {
	Schema     string      `json:"schema"`
	ModelRef   modelRef    `json:"model"`
	Frame      frameRef    `json:"frame"`
	Detections []Detection `json:"persons"`

	skeleton [][2]int
}

// Family returns the result family.
func (r *KeypointResult) Family() string { return "pose" }

// JSON returns the structured payload.
func (r *KeypointResult) JSON() any { return r }

// Readings emits person count and a bounded keypoint JSON point.
func (r *KeypointResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	return []Reading{
		{Name: "PersonCount", Type: "int32", Value: safeInt32(len(r.Detections))},
		{Name: "KeypointsJSON", Type: "string", Value: payload},
	}, nil
}

// Draw renders boxes, point markers and skeleton edges.
func (r *KeypointResult) Draw(canvas *image.RGBA, _ []string) {
	for _, d := range r.Detections {
		drawBox(canvas, d.Box, d.Class, d.Label, d.Score)
		for _, kp := range d.Keypoints {
			if kp.Score <= 0 {
				continue
			}
			b := canvas.Bounds()
			x := int(kp.X * float32(b.Dx()))
			y := int(kp.Y * float32(b.Dy()))
			fillRect(canvas, x-1, y-1, x+2, y+2, palette(d.Class+2))
		}
		for _, e := range r.skeleton {
			if e[0] >= len(d.Keypoints) || e[1] >= len(d.Keypoints) {
				continue
			}
			a, z := d.Keypoints[e[0]], d.Keypoints[e[1]]
			if a.Score <= 0 || z.Score <= 0 {
				continue
			}
			drawLine(canvas, a, z)
		}
	}
}

type keypointsSpec struct {
	Output     string  `json:"output"`
	Threshold  float32 `json:"threshold"`
	NMS        nmsSpec `json:"nms"`
	Start      int     `json:"start"`
	Stride     int     `json:"stride"`
	Count      int     `json:"count"`
	KPScoreMin float32 `json:"kp_score_min"`
}

type keypointsDecoder struct{ spec keypointsSpec }

func newKeypointsDecoder(raw json.RawMessage) (Decoder, error) {
	d := &keypointsDecoder{spec: keypointsSpec{
		Output: "", Threshold: 0.25, NMS: nmsSpec{IoU: 0.45, TopK: 300},
		Start: 5, Stride: 3, Count: 17, KPScoreMin: 0.5,
	}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *keypointsDecoder) Kind() string { return "keypoints" }

// Decode parses YOLOv8-pose style [1,4+1+k*stride,A] tensors.
func (d *keypointsDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 3 || t.Shape[0] != 1 {
		return nil, badShape("keypoints", "[1,4+1+k*stride,A]", t.Shape)
	}
	channels, anchors := int(t.Shape[1]), int(t.Shape[2])
	if channels <= 0 || anchors <= 0 || d.spec.Start < 0 || d.spec.Count <= 0 || d.spec.Stride < 3 {
		return nil, badShape("keypoints", "positive tensor dimensions and x,y,score keypoint stride", t.Shape)
	}
	count64, stride64 := int64(d.spec.Count), int64(d.spec.Stride)
	if count64 > (math.MaxInt64-int64(d.spec.Start))/stride64 {
		return nil, shapeErr("keypoint layout count=%d stride=%d overflows", d.spec.Count, d.spec.Stride)
	}
	expected64 := int64(d.spec.Start) + count64*stride64
	if expected64 > math.MaxInt32 || int64(channels) < expected64 {
		return nil, shapeErr("keypoints channels %d < required %d", channels, expected64)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, fmt.Errorf("keypoints tensor truncated or too large")
	}
	type cand struct {
		det Detection
	}
	var cands []cand
	scores := make([]float32, 0)
	classes := make([]int, 0)
	boxes := make([][4]float32, 0)
	threshold := in.Threshold(d.spec.Threshold)
	for a := 0; a < anchors; a++ {
		score := t.Float32[4*anchors+a]
		cx, cy, w, h := t.Float32[a], t.Float32[anchors+a], t.Float32[2*anchors+a], t.Float32[3*anchors+a]
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) ||
			score < threshold || !finiteBox(cx, cy, w, h) {
			continue
		}
		box := normalizeAndUnmap(in.Meta, cx-w/2, cy-h/2, cx+w/2, cy+h/2, false)
		det := Detection{Class: 0, Label: "person", Score: score, Box: box}
		for k := 0; k < d.spec.Count; k++ {
			base := (d.spec.Start+k*d.spec.Stride)*anchors + a
			x, y, s := t.Float32[base], t.Float32[base+1], t.Float32[base+2]
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) ||
				s < d.spec.KPScoreMin || !finiteBox(x, y, s, 0) {
				det.Keypoints = append(det.Keypoints, Keypoint{Score: 0})
				continue
			}
			u, v, _, _ := in.Meta.UnmapBox(x/float32(in.Meta.InputW), y/float32(in.Meta.InputH),
				x/float32(in.Meta.InputW), y/float32(in.Meta.InputH))
			det.Keypoints = append(det.Keypoints, Keypoint{X: u, Y: v, Score: s})
		}
		cands = append(cands, cand{det: det})
		scores = append(scores, score)
		classes = append(classes, 0)
		boxes = append(boxes, box)
	}
	kept := nonMaxSuppressIndices(len(cands),
		func(i int) float32 { return scores[i] },
		func(i int) int { return classes[i] },
		func(i int) [4]float32 { return boxes[i] },
		d.spec.NMS.orDefault())
	dets := make([]Detection, 0, len(kept))
	for _, i := range kept {
		dets = append(dets, cands[i].det)
	}
	return &KeypointResult{
		Schema:     "nereid.pose/1",
		ModelRef:   modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:      frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		Detections: dets,
		skeleton:   defaultSkeleton(d.spec.Count),
	}, nil
}

// defaultSkeleton returns the COCO-17 edge table; other keypoint sets are
// descriptor-driven in later releases via names/skeleton files.
func defaultSkeleton(count int) [][2]int {
	if count == 17 {
		return [][2]int{
			{0, 1}, {0, 2}, {1, 3}, {2, 4},
			{5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
			{5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
		}
	}
	return nil
}
