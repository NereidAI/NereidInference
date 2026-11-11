// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

// ---- segmentation.semantic ([1,nc,H,W] argmax) ----

type semanticSpec struct {
	Output string  `json:"output"`
	Alpha  float32 `json:"overlay_alpha"`
}

type semanticDecoder struct{ spec semanticSpec }

func newSemanticSegDecoder(raw json.RawMessage) (Decoder, error) {
	d := &semanticDecoder{spec: semanticSpec{Alpha: 0.4}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *semanticDecoder) Kind() string { return "segmentation.semantic" }

// ClassRegion summarizes one class: area fraction and bounding box.
type ClassRegion struct {
	Class int        `json:"cls"`
	Label string     `json:"label"`
	Area  float32    `json:"area"`
	Box   [4]float32 `json:"box"`
}

// SegmentationResult covers semantic and instance outputs.
type SegmentationResult struct {
	Schema   string         `json:"schema"`
	ModelRef modelRef       `json:"model"`
	Frame    frameRef       `json:"frame"`
	Regions  []ClassRegion  `json:"regions,omitempty"`
	Masks    [][][2]float32 `json:"polygons,omitempty"`

	family        string
	jsonName      string
	labels        []string
	classMap      []byte // semantic: HxW class ids, for preview
	overlayAlpha  float32
	mh, mw        int
	instanceMasks [][]bool // instance: mh*mw masks
	instanceBoxes [][4]float32
}

// Family returns the result family.
func (r *SegmentationResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *SegmentationResult) JSON() any { return r }

// Readings emits class count and a bounded polygon/region JSON point.
func (r *SegmentationResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	count := safeInt32(len(r.Regions))
	return []Reading{
		{Name: "SegmentCount", Type: "int32", Value: count},
		{Name: r.jsonName, Type: "string", Value: payload},
	}, nil
}

// Draw overlays translucent class colors (semantic) or mask polygons.
func (r *SegmentationResult) Draw(canvas *image.RGBA, _ []string) {
	switch r.family {
	case "segmentation":
		drawClassMap(canvas, r.classMap, r.mh, r.mw, r.specAlpha())
	default:
		for i, mask := range r.instanceMasks {
			drawMask(canvas, mask, r.mh, r.mw, r.instanceBoxes[i], palette(i+1))
		}
	}
}

func (r *SegmentationResult) specAlpha() float32 {
	if r.overlayAlpha <= 0 {
		return 0.4
	}
	return r.overlayAlpha
}

// Decode computes argmax regions from a [1,nc,H,W] logits tensor.
func (d *semanticDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 4 || t.Shape[0] != 1 {
		return nil, badShape("segmentation.semantic", "[1,nc,H,W]", t.Shape)
	}
	nc, mh, mw := int(t.Shape[1]), int(t.Shape[2]), int(t.Shape[3])
	if nc <= 0 || mh <= 0 || mw <= 0 {
		return nil, badShape("segmentation.semantic", "positive [1,nc,H,W]", t.Shape)
	}
	if nc > 255 {
		return nil, fmt.Errorf("segmentation.semantic supports at most 255 classes, got %d", nc)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, fmt.Errorf("segmentation.semantic tensor truncated or too large")
	}
	maskSize, ok := shapeProduct([]int64{int64(mh), int64(mw)})
	if !ok {
		return nil, badShape("segmentation.semantic", "addressable HxW mask", t.Shape)
	}
	classMap := make([]byte, maskSize)
	type acc struct {
		area           int
		x0, y0, x1, y1 int
	}
	accs := make([]acc, nc)
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			best, bestV := 0, float32(math.Inf(-1))
			for c := 0; c < nc; c++ {
				v := t.Float32[(c*mh+y)*mw+x]
				if v > bestV {
					bestV, best = v, c
				}
			}
			idx := y*mw + x
			classMap[idx] = byte(best)
			a := &accs[best]
			a.area++
			if a.area == 1 {
				a.x0, a.y0, a.x1, a.y1 = x, y, x, y
			} else {
				if x < a.x0 {
					a.x0 = x
				}
				if y < a.y0 {
					a.y0 = y
				}
				if x > a.x1 {
					a.x1 = x
				}
				if y > a.y1 {
					a.y1 = y
				}
			}
		}
	}
	total := float32(mh * mw)
	var regions []ClassRegion
	for c := 0; c < nc; c++ {
		if accs[c].area == 0 {
			continue
		}
		a := accs[c]
		x1, y1, x2, y2 := in.Meta.UnmapBox(
			float32(a.x0)/float32(mw), float32(a.y0)/float32(mh),
			float32(a.x1+1)/float32(mw), float32(a.y1+1)/float32(mh),
		)
		regions = append(regions, ClassRegion{
			Class: c,
			Label: labelFor(in.Labels, c),
			Area:  float32(a.area) / total,
			Box:   [4]float32{x1, y1, x2, y2},
		})
	}
	return &SegmentationResult{
		Schema:       "nereid.segmentation/1",
		ModelRef:     modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:        frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		Regions:      regions,
		family:       "segmentation",
		jsonName:     "SegmentationJSON",
		labels:       in.Labels,
		classMap:     classMap,
		overlayAlpha: d.spec.Alpha,
		mh:           mh, mw: mw,
	}, nil
}

// ---- segmentation.instance (YOLOv8-seg predictions + mask protos) ----

const (
	maxMaskDim            = 4096
	maxMaskPrototypes     = 256
	maxInstanceMaskPixels = 16_000_000
	maxMaskCompute        = 512_000_000
)

type instanceSpec struct {
	Predictions   string  `json:"predictions"`
	Protos        string  `json:"mask_protos"`
	Threshold     float32 `json:"threshold"`
	NMS           nmsSpec `json:"nms"`
	MaskThreshold float32 `json:"mask_threshold"`
	MaxPoints     int     `json:"max_points"`
}

type instanceDecoder struct{ spec instanceSpec }

func newInstanceSegDecoder(raw json.RawMessage) (Decoder, error) {
	d := &instanceDecoder{spec: instanceSpec{
		Predictions: "predictions", Protos: "mask_protos",
		Threshold: 0.25, MaskThreshold: 0.5, NMS: nmsSpec{IoU: 0.45, TopK: 300},
		MaxPoints: 100,
	}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *instanceDecoder) Kind() string { return "segmentation.instance" }

// Decode performs box NMS then reconstructs per-instance masks through the
// coefficient x prototype product.
func (d *instanceDecoder) Decode(in *Input) (Result, error) {
	pred, err := outputTensor(in, d.spec.Predictions)
	if err != nil {
		return nil, err
	}
	protos, err := outputTensor(in, d.spec.Protos)
	if err != nil {
		return nil, err
	}
	if len(pred.Shape) != 3 || pred.Shape[0] != 1 {
		return nil, badShape("segmentation.instance", "[1,4+nc+nm,A]", pred.Shape)
	}
	if len(protos.Shape) != 4 || protos.Shape[0] != 1 {
		return nil, badShape("mask_protos", "[1,nm,mH,mW]", protos.Shape)
	}
	channels, anchors := int(pred.Shape[1]), int(pred.Shape[2])
	nm, mh, mw := int(protos.Shape[1]), int(protos.Shape[2]), int(protos.Shape[3])
	if channels <= 0 || anchors <= 0 || nm <= 0 || mh <= 0 || mw <= 0 {
		return nil, badShape("segmentation.instance", "positive prediction and prototype dimensions", append(append([]int64{}, pred.Shape...), protos.Shape...))
	}
	if nm > maxMaskPrototypes || mh > maxMaskDim || mw > maxMaskDim {
		return nil, badShape("segmentation.instance", "mask dimensions within decoder limits", protos.Shape)
	}
	predRequired, ok := shapeProduct(pred.Shape)
	if !ok || len(pred.Float32) < predRequired {
		return nil, fmt.Errorf("segmentation.instance predictions tensor truncated or too large")
	}
	protoRequired, ok := shapeProduct(protos.Shape)
	if !ok || len(protos.Float32) < protoRequired {
		return nil, fmt.Errorf("segmentation.instance prototype tensor truncated or too large")
	}
	maskSize, ok := shapeProduct([]int64{int64(mh), int64(mw)})
	if !ok {
		return nil, badShape("segmentation.instance", "addressable mask HxW", protos.Shape)
	}
	nc := channels - 4 - nm
	if nc <= 0 {
		return nil, shapeErr("instance channels %d leave no classes", channels)
	}
	type candidate struct {
		det   Detection
		coeff []float32
	}
	var cands []candidate
	for a := 0; a < anchors; a++ {
		bestClass, bestScore := -1, in.Threshold(d.spec.Threshold)
		for c := 0; c < nc; c++ {
			if s := pred.Float32[(4+c)*anchors+a]; s > bestScore {
				bestScore, bestClass = s, c
			}
		}
		if bestClass < 0 {
			continue
		}
		cx, cy, w, h := pred.Float32[a], pred.Float32[anchors+a], pred.Float32[2*anchors+a], pred.Float32[3*anchors+a]
		if !finiteBox(cx, cy, w, h) {
			continue
		}
		box := normalizeAndUnmap(in.Meta, cx-w/2, cy-h/2, cx+w/2, cy+h/2, false)
		cv := make([]float32, nm)
		for k := 0; k < nm; k++ {
			cv[k] = pred.Float32[(4+nc+k)*anchors+a]
		}
		cands = append(cands, candidate{
			det:   Detection{Class: bestClass, Label: labelFor(in.Labels, bestClass), Score: bestScore, Box: box},
			coeff: cv,
		})
	}
	kept := nonMaxSuppressIndices(len(cands),
		func(i int) float32 { return cands[i].det.Score },
		func(i int) int { return cands[i].det.Class },
		func(i int) [4]float32 { return cands[i].det.Box },
		d.spec.NMS.orDefault())
	if len(kept) > 0 {
		maskBudget, ok := shapeProduct([]int64{int64(len(kept)), int64(mh), int64(mw)})
		if !ok || maskBudget > maxInstanceMaskPixels ||
			int64(len(kept))*int64(nm)*int64(maskSize) > maxMaskCompute {
			return nil, fmt.Errorf("segmentation.instance mask reconstruction exceeds decoder budget (%d detections at %dx%d with %d prototypes)", len(kept), mh, mw, nm)
		}
	}
	result := &SegmentationResult{
		Schema:   "nereid.instance-segmentation/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Frame:    frameRef{Width: in.Meta.SourceW, Height: in.Meta.SourceH},
		family:   "instance-segmentation",
		jsonName: "InstanceJSON",
		labels:   in.Labels,
		mh:       mh, mw: mw,
	}
	for _, idx := range kept {
		mask := make([]bool, maskSize)
		for p := 0; p < maskSize; p++ {
			var dot float32
			for k := 0; k < nm; k++ {
				dot += cands[idx].coeff[k] * protos.Float32[k*mh*mw+p]
			}
			mask[p] = sigmoid(dot) > d.spec.MaskThreshold
		}
		cropToBox(mask, mh, mw, cands[idx].det.Box, in.Meta)
		poly := boundaryPolygon(mask, mh, mw, d.spec.MaxPoints, in.Meta)
		result.Masks = append(result.Masks, poly)
		result.instanceMasks = append(result.instanceMasks, mask)
		result.instanceBoxes = append(result.instanceBoxes, cands[idx].det.Box)
		result.Regions = append(result.Regions, ClassRegion{
			Class: cands[idx].det.Class, Label: cands[idx].det.Label, Box: cands[idx].det.Box,
		})
	}
	return result, nil
}

func drawClassMap(canvas *image.RGBA, classMap []byte, mh, mw int, alpha float32) {
	b := canvas.Bounds()
	sx := float64(b.Dx()) / float64(mw)
	sy := float64(b.Dy()) / float64(mh)
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			cid := int(classMap[y*mw+x])
			c := palette(cid)
			x0 := b.Min.X + int(float64(x)*sx)
			y0 := b.Min.Y + int(float64(y)*sy)
			x1 := b.Min.X + int(float64(x+1)*sx) - 1
			y1 := b.Min.Y + int(float64(y+1)*sy) - 1
			fillCell(canvas, x0, y0, x1, y1, func(ix, iy int) {
				blendPixel(canvas, ix, iy, color.RGBA{R: c.R, G: c.G, B: c.B, A: 255}, alpha)
			})
		}
	}
}

func drawMask(canvas *image.RGBA, mask []bool, mh, mw int, box [4]float32, c color.Color) {
	b := canvas.Bounds()
	sx := float64(b.Dx()) / float64(mw)
	sy := float64(b.Dy()) / float64(mh)
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			if !mask[y*mw+x] {
				continue
			}
			x0 := b.Min.X + int(float64(x)*sx)
			y0 := b.Min.Y + int(float64(y)*sy)
			x1 := b.Min.X + int(float64(x+1)*sx) - 1
			y1 := b.Min.Y + int(float64(y+1)*sy) - 1
			fillCell(canvas, x0, y0, x1, y1, func(ix, iy int) {
				blendPixel(canvas, ix, iy, c, 0.5)
			})
		}
	}
}

func blendPixel(img *image.RGBA, x, y int, c color.Color, alpha float32) {
	r, g, b, _ := c.RGBA()
	off := img.PixOffset(x, y)
	if off < 0 || off >= len(img.Pix) {
		return
	}
	or, og, ob := img.Pix[off], img.Pix[off+1], img.Pix[off+2]
	img.Pix[off] = uint8(float32(or)*(1-alpha) + float32(r>>8)*alpha)
	img.Pix[off+1] = uint8(float32(og)*(1-alpha) + float32(g>>8)*alpha)
	img.Pix[off+2] = uint8(float32(ob)*(1-alpha) + float32(b>>8)*alpha)
}

func badShape(kind, want string, got []int64) error {
	return fmt.Errorf("%s expects %s, got %v", kind, want, got)
}

func shapeErr(format string, args ...any) error {
	return fmt.Errorf(format, args...)
}

// cropToBox removes mask pixels outside the detection box. Mask pixels are
// addressed in normalized model-input space and mapped to source-frame
// coordinates with the preprocess metadata.
func cropToBox(mask []bool, mh, mw int, box [4]float32, meta preprocess.Meta) {
	for y := 0; y < mh; y++ {
		for x := 0; x < mw; x++ {
			nx := float32(x) / float32(mw)
			ny := float32(y) / float32(mh)
			ux, uy, _, _ := meta.UnmapBox(nx, ny, nx, ny)
			if ux < box[0] || ux > box[2] || uy < box[1] || uy > box[3] {
				mask[y*mw+x] = false
			}
		}
	}
}
