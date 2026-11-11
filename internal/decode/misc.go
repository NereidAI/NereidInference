// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"unicode/utf8"
)

// ---- embeddings.vector ----

type embeddingSpec struct {
	Output    string `json:"output"`
	Normalize bool   `json:"l2_normalize"`
	MaxDims   int    `json:"max_dims"`
}

type embeddingDecoder struct{ spec embeddingSpec }

func newEmbeddingDecoder(raw json.RawMessage) (Decoder, error) {
	d := &embeddingDecoder{spec: embeddingSpec{Normalize: true, MaxDims: 512}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *embeddingDecoder) Kind() string { return "embeddings.vector" }

// VectorResult is a normalized feature vector (face/ReID); matching itself
// is a business application concern and never lives in this service.
type VectorResult struct {
	Schema   string    `json:"schema"`
	ModelRef modelRef  `json:"model"`
	Vector   []float32 `json:"vector"`
	family   string
}

// Family returns the result family.
func (r *VectorResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *VectorResult) JSON() any { return r }

// Readings emits a bounded vector JSON point.
func (r *VectorResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	return []Reading{{Name: "VectorJSON", Type: "string", Value: payload}}, nil
}

// Draw is a no-op for embeddings.
func (r *VectorResult) Draw(*image.RGBA, []string) {}

// Decode extracts [1,D] (or [D]) vectors with optional L2 normalization.
func (d *embeddingDecoder) Decode(in *Input) (Result, error) {
	var v []float32
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		if len(in.Response.Embeddings) == 0 || len(in.Response.Embeddings[0]) == 0 {
			return nil, err
		}
		v = append([]float32(nil), in.Response.Embeddings[0]...)
	} else {
		var dims int
		switch len(t.Shape) {
		case 1:
			dims = int(t.Shape[0])
		case 2:
			if t.Shape[0] != 1 {
				return nil, badShape("embeddings.vector", "[D] or [1,D]", t.Shape)
			}
			dims = int(t.Shape[1])
		default:
			return nil, badShape("embeddings.vector", "[D] or [1,D]", t.Shape)
		}
		if dims <= 0 || len(t.Float32) < dims {
			return nil, badShape("embeddings.vector", "complete [D] or [1,D]", t.Shape)
		}
		v = append([]float32(nil), t.Float32[:dims]...)
	}
	if d.spec.Normalize {
		var n float32
		for _, x := range v {
			n += x * x
		}
		if n > 0 {
			n = float32(math.Sqrt(float64(n)))
			for i := range v {
				v[i] /= n
			}
		}
	}
	if d.spec.MaxDims > 0 && len(v) > d.spec.MaxDims {
		v = v[:d.spec.MaxDims]
	}
	return &VectorResult{
		Schema:   "nereid.embeddings/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Vector:   v,
		family:   "embeddings",
	}, nil
}

// ---- dense.map (count/depth/anomaly/optical-flow/alpha) ----

type denseSpec struct {
	Output string `json:"output"`
	Mode   string `json:"mode"` // sum|depth|flow|mask
}

type denseDecoder struct{ spec denseSpec }

func newDenseMapDecoder(raw json.RawMessage) (Decoder, error) {
	d := &denseDecoder{spec: denseSpec{Mode: "sum"}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *denseDecoder) Kind() string { return "dense.map" }

// DenseResult carries a scalar aggregate plus the map for preview.
type DenseResult struct {
	Schema   string             `json:"schema"`
	ModelRef modelRef           `json:"model"`
	Stats    map[string]float64 `json:"stats"`

	family string
	mode   string
	map1   []float32
	map2   []float32
	mh, mw int
}

// Family returns the result family.
func (r *DenseResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *DenseResult) JSON() any { return r }

// Readings emits the aggregate scalar and a stats JSON point.
func (r *DenseResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	name := "DenseValue"
	v := r.Stats["sum"]
	if r.mode == "depth" {
		name, v = "DepthMean", r.Stats["mean"]
	}
	return []Reading{
		{Name: name, Type: "float64", Value: v},
		{Name: "DenseJSON", Type: "string", Value: payload},
	}, nil
}

// Draw renders a pseudo-color heat map overlay.
func (r *DenseResult) Draw(canvas *image.RGBA, _ []string) {
	b := canvas.Bounds()
	sx := float64(b.Dx()) / float64(r.mw)
	sy := float64(b.Dy()) / float64(r.mh)
	var lo, hi = r.Stats["min"], r.Stats["max"]
	if hi-lo < 1e-6 {
		hi = lo + 1
	}
	for y := 0; y < r.mh; y++ {
		for x := 0; x < r.mw; x++ {
			t01 := (float64(r.map1[y*r.mw+x]) - lo) / (hi - lo)
			if t01 < 0 {
				t01 = 0
			}
			if t01 > 1 {
				t01 = 1
			}
			x0 := b.Min.X + int(float64(x)*sx)
			y0 := b.Min.Y + int(float64(y)*sy)
			x1 := b.Min.X + int(float64(x+1)*sx) - 1
			y1 := b.Min.Y + int(float64(y+1)*sy) - 1
			fillCell(canvas, x0, y0, x1, y1, func(ix, iy int) {
				heat(canvas, ix, iy, t01)
			})
		}
	}
}

// Decode reduces one- or two-channel spatial tensors.
func (d *denseDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 4 || t.Shape[0] != 1 {
		return nil, badShape("dense.map", "[1,1|2,H,W]", t.Shape)
	}
	ch, mh, mw := int(t.Shape[1]), int(t.Shape[2]), int(t.Shape[3])
	if ch < 1 || ch > 2 || mh <= 0 || mw <= 0 {
		return nil, badShape("dense.map", "positive [1,1|2,H,W]", t.Shape)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, badShape("dense.map", "complete [1,1|2,H,W]", t.Shape)
	}
	plane, ok := shapeProduct([]int64{int64(mh), int64(mw)})
	if !ok {
		return nil, badShape("dense.map", "addressable HxW", t.Shape)
	}
	m1 := t.Float32[:plane]
	var sum, mn, mx float64
	mn = math.MaxFloat64
	for _, v := range m1 {
		f := float64(v)
		sum += f
		if f < mn {
			mn = f
		}
		if f > mx {
			mx = f
		}
	}
	res := &DenseResult{
		Schema:   "nereid.dense/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Stats: map[string]float64{
			"sum": sum, "mean": sum / float64(plane), "min": mn, "max": mx,
		},
		family: "dense", mode: d.spec.Mode, map1: m1, mh: mh, mw: mw,
	}
	if ch >= 2 {
		res.map2 = t.Float32[plane : 2*plane]
	}
	return res, nil
}

func heat(img *image.RGBA, x, y int, t float64) {
	r := uint8(math.Min(255, math.Max(0, 255*t*1.5)))
	g := uint8(math.Min(255, math.Max(0, 255*(1-math.Abs(2*t-1)))))
	b := uint8(math.Min(255, math.Max(0, 255*(1-t)*1.5)))
	blendPixel(img, x, y, color.RGBA{R: r, G: g, B: b, A: 255}, 0.4)
}

// ---- image.tensor (super-resolution / enhancement) ----

type imageTensorSpec struct {
	Output  string  `json:"output"`
	Mean    float32 `json:"mean"`
	Scale   float32 `json:"scale"`
	Quality int     `json:"jpeg_quality"`
}

type imageTensorDecoder struct{ spec imageTensorSpec }

func newImageTensorDecoder(raw json.RawMessage) (Decoder, error) {
	d := &imageTensorDecoder{spec: imageTensorSpec{Scale: 1, Quality: 85}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	if d.spec.Quality < 1 || d.spec.Quality > 100 {
		return nil, fmt.Errorf("image.tensor jpeg_quality must be in 1..100")
	}
	return d, nil
}

func (d *imageTensorDecoder) Kind() string { return "image.tensor" }

// ImageResult carries the produced JPEG for preview/blob readings.
type ImageResult struct {
	Schema   string   `json:"schema"`
	ModelRef modelRef `json:"model"`
	JPEG     []byte   `json:"-"`
	family   string
}

// Family returns the result family.
func (r *ImageResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *ImageResult) JSON() any {
	return map[string]any{"schema": r.Schema, "model": r.ModelRef, "jpeg_bytes": len(r.JPEG)}
}

// Readings emits one binary JPEG reading.
func (r *ImageResult) Readings(_ int) ([]Reading, error) {
	return []Reading{{Name: "ResultJPEG", Type: "binary", Value: r.JPEG}}, nil
}

// Draw replaces the canvas with the output image.
func (r *ImageResult) Draw(canvas *image.RGBA, _ []string) {
	img, err := jpeg.Decode(bytes.NewReader(r.JPEG))
	if err != nil {
		return
	}
	b := canvas.Bounds()
	dst := image.NewRGBA(b)
	sb := img.Bounds()
	for y := 0; y < b.Dy(); y++ {
		sy := sb.Min.Y + y*sb.Dy()/b.Dy()
		for x := 0; x < b.Dx(); x++ {
			sx := sb.Min.X + x*sb.Dx()/b.Dx()
			dst.Set(x, y, img.At(sx, sy))
		}
	}
	copy(canvas.Pix, dst.Pix)
}

// Decode converts [1,3,H,W] float tensor back to a JPEG image.
func (d *imageTensorDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 4 || t.Shape[0] != 1 || t.Shape[1] != 3 {
		return nil, badShape("image.tensor", "[1,3,H,W]", t.Shape)
	}
	h, w := int(t.Shape[2]), int(t.Shape[3])
	if h <= 0 || w <= 0 {
		return nil, badShape("image.tensor", "positive H,W", t.Shape)
	}
	const maxImageDim = 4096
	if h > maxImageDim || w > maxImageDim {
		return nil, badShape("image.tensor", "dimensions at most 4096", t.Shape)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, badShape("image.tensor", "complete [1,3,H,W]", t.Shape)
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	plane, ok := shapeProduct([]int64{int64(h), int64(w)})
	if !ok {
		return nil, badShape("image.tensor", "addressable HxW", t.Shape)
	}
	for i := 0; i < plane; i++ {
		r8 := clampU8(t.Float32[i]*d.spec.Scale + d.spec.Mean)
		g8 := clampU8(t.Float32[plane+i]*d.spec.Scale + d.spec.Mean)
		b8 := clampU8(t.Float32[2*plane+i]*d.spec.Scale + d.spec.Mean)
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = r8, g8, b8, 255
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: d.spec.Quality}); err != nil {
		return nil, err
	}
	return &ImageResult{
		Schema:   "nereid.image/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		JPEG:     buf.Bytes(),
		family:   "image",
	}, nil
}

func clampU8(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// ---- sequence.ctc (OCR recognition head) ----

type ctcSpec struct {
	Output     string `json:"output"`
	BlankIndex int    `json:"blank_index"`
	Activation string `json:"activation"` // softmax (default) or none
}

type ctcDecoder struct{ spec ctcSpec }

func newCTCDecoder(raw json.RawMessage) (Decoder, error) {
	d := &ctcDecoder{spec: ctcSpec{BlankIndex: 0}}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *ctcDecoder) Kind() string { return "sequence.ctc" }

// TextResult is used by CTC and passthrough text outputs.
type TextResult struct {
	Schema     string   `json:"schema"`
	ModelRef   modelRef `json:"model"`
	Text       string   `json:"text"`
	Confidence float32  `json:"confidence"`
	family     string
	jsonName   string
}

// Family returns the result family.
func (r *TextResult) Family() string { return r.family }

// JSON returns the structured payload.
func (r *TextResult) JSON() any { return r }

// Readings emits text and confidence points.
func (r *TextResult) Readings(capBytes int) ([]Reading, error) {
	text := r.Text
	if capBytes > 0 && len(text) > capBytes {
		text = truncateUTF8(text, capBytes)
	}
	return []Reading{
		{Name: r.jsonName, Type: "string", Value: text},
		{Name: "Confidence", Type: "float64", Value: float64(r.Confidence)},
	}, nil
}

func truncateUTF8(s string, max int) string {
	if len(s) <= max {
		return s
	}
	for len(s) > max {
		_, size := utf8.DecodeLastRuneInString(s)
		if size == 0 || size >= len(s) {
			break
		}
		s = s[:len(s)-size]
	}
	return s
}

// Draw paints the recognized text.
func (r *TextResult) Draw(canvas *image.RGBA, _ []string) { caption(canvas, 4, 14, r.Text) }

// Decode performs CTC greedy collapse on a [T,C] matrix; labels file lines
// are the character table.
func (d *ctcDecoder) Decode(in *Input) (Result, error) {
	t, err := outputTensor(in, d.spec.Output)
	if err != nil {
		return nil, err
	}
	var timeSteps, classes int
	switch {
	case len(t.Shape) == 2:
		timeSteps, classes = int(t.Shape[0]), int(t.Shape[1])
	case len(t.Shape) == 3 && t.Shape[0] == 1:
		timeSteps, classes = int(t.Shape[1]), int(t.Shape[2])
	default:
		return nil, badShape("sequence.ctc", "[T,C] or [1,T,C]", t.Shape)
	}
	if timeSteps <= 0 || classes <= 0 {
		return nil, badShape("sequence.ctc", "positive [T,C] or [1,T,C]", t.Shape)
	}
	required, ok := shapeProduct(t.Shape)
	if !ok || len(t.Float32) < required {
		return nil, badShape("sequence.ctc", "complete [T,C] or [1,T,C]", t.Shape)
	}
	activation := d.spec.Activation
	if activation == "" {
		activation = "softmax"
	}
	if activation != "softmax" && activation != "none" {
		return nil, fmt.Errorf("sequence.ctc activation must be softmax or none, got %q", activation)
	}
	var runes []rune
	var conf float32
	var confN int
	prev := -1
	for ti := 0; ti < timeSteps; ti++ {
		row := t.Float32[ti*classes : (ti+1)*classes]
		best, bestV := 0, float32(math.Inf(-1))
		if activation == "softmax" {
			var maxV float32
			for _, v := range row {
				if v > maxV {
					maxV = v
				}
			}
			var sum float32
			probs := make([]float32, classes)
			for i, v := range row {
				p := float32(math.Exp(float64(v - maxV)))
				probs[i] = p
				sum += p
			}
			if sum > 0 {
				for i := range probs {
					probs[i] /= sum
				}
			}
			row = probs
		}
		for c, v := range row {
			if v > bestV {
				best, bestV = c, v
			}
		}
		if best != d.spec.BlankIndex && best != prev && best < len(in.Labels) {
			runes = append(runes, []rune(in.Labels[best])...)
			conf += bestV
			confN++
		}
		prev = best
	}
	if confN > 0 {
		conf /= float32(confN)
	}
	return &TextResult{
		Schema: "nereid.ctc/1", ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Text: string(runes), Confidence: conf, family: "ocr", jsonName: "Text",
	}, nil
}

// ---- composite (multiple named member results, e.g. ANPR) ----

type compositeMember struct {
	Name   string          `json:"name"`
	Kind   string          `json:"kind"`
	Config json.RawMessage `json:"config"`
}

type compositeSpec struct {
	Members []compositeMember `json:"members"`
}

type compositeDecoder struct{ spec compositeSpec }

func newCompositeDecoder(raw json.RawMessage) (Decoder, error) {
	d := &compositeDecoder{}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &d.spec); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *compositeDecoder) Kind() string { return "composite" }

// CompositeResult merges member results for compound heads.
type CompositeResult struct {
	Schema   string         `json:"schema"`
	ModelRef modelRef       `json:"model"`
	Members  map[string]any `json:"members"`
	memberR  []Result
}

// Family returns the result family.
func (r *CompositeResult) Family() string { return "composite" }

// JSON returns the structured payload.
func (r *CompositeResult) JSON() any { return r }

// Readings unions member readings under one composite JSON point.
func (r *CompositeResult) Readings(capBytes int) ([]Reading, error) {
	payload, err := marshalBounded(r, capBytes)
	if err != nil {
		return nil, err
	}
	return []Reading{{Name: "CompositeJSON", Type: "string", Value: payload}}, nil
}

// Draw renders every member overlay.
func (r *CompositeResult) Draw(canvas *image.RGBA, labels []string) {
	for _, m := range r.memberR {
		m.Draw(canvas, labels)
	}
}

// Decode runs each member codec against the same response.
func (d *compositeDecoder) Decode(in *Input) (Result, error) {
	res := &CompositeResult{
		Schema:   "nereid.composite/1",
		ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Members:  map[string]any{},
	}
	for _, m := range d.spec.Members {
		factory, ok := defaultFactories[m.Kind]
		if !ok {
			return nil, fmt.Errorf("composite member %q: decoder kind %q is not registered", m.Name, m.Kind)
		}
		if len(m.Config) == 0 {
			return nil, fmt.Errorf("composite member %q requires its own decoder config", m.Name)
		}
		dec, err := factory(m.Config)
		if err != nil {
			return nil, err
		}
		r, err := dec.Decode(in)
		if err != nil {
			return nil, err
		}
		res.Members[m.Name] = r.JSON()
		res.memberR = append(res.memberR, r)
	}
	return res, nil
}

// ---- passthrough.text (OpenAI chat / embeddings textual response) ----

type passthroughDecoder struct{}

func newPassthroughDecoder(json.RawMessage) (Decoder, error) { return &passthroughDecoder{}, nil }
func (d *passthroughDecoder) Kind() string                   { return "passthrough.text" }

// Decode wraps the backend response text.
func (d *passthroughDecoder) Decode(in *Input) (Result, error) {
	return &TextResult{
		Schema: "nereid.text/1", ModelRef: modelRef{ID: in.Manifest.Model.ID, Version: in.Manifest.Model.Version},
		Text: in.Response.Text, family: "text", jsonName: "Text",
	}, nil
}
