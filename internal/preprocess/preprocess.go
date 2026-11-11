// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package preprocess turns acquired frames into model input tensors using
// only the descriptor's io.inputs[].preprocess parameters. It is pure Go
// (no CGo/OpenCV): JPEG decode is from the standard library and resize is
// bilinear. Coordinate-unmap metadata produced here lets decoders return
// boxes in the source-frame coordinate system.
package preprocess

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"strconv"
	"strings"

	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/model"
)

// Meta carries the geometry needed to map model-space coordinates back to
// the source frame.
type Meta struct {
	// Scale is retained for uniform resize paths (letterbox/crop).
	Scale float32
	// ScaleX/ScaleY map source pixels to model pixels. They differ under
	// stretch resize when source/model aspect ratios are not equal.
	ScaleX float32
	ScaleY float32
	// PadX/PadY are padding offsets (in model pixels) after letterboxing.
	// Crop stores negative offsets because the model window is inside image.
	PadX float32
	PadY float32
	// InputW/InputH are the model input tensor spatial dimensions.
	InputW int
	InputH int
	// SourceW/SourceH are the acquired frame dimensions.
	SourceW int
	SourceH int
}

// UnmapPoint converts one normalized model-input point to normalized source
// coordinates without clipping. Alignment needs the original geometry even
// when a predicted landmark lies just outside the frame.
func (m Meta) UnmapPoint(x, y float32) (float32, float32) {
	sx, sy := m.ScaleX, m.ScaleY
	if sx == 0 {
		sx = m.Scale
	}
	if sy == 0 {
		sy = m.Scale
	}
	return (x*float32(m.InputW) - m.PadX) / (sx * float32(m.SourceW)),
		(y*float32(m.InputH) - m.PadY) / (sy * float32(m.SourceH))
}

// UnmapBox converts a normalized [0,1] box in model-input space back to
// clipped normalized coordinates in the source frame.
func (m Meta) UnmapBox(x1, y1, x2, y2 float32) (float32, float32, float32, float32) {
	x1, y1 = m.UnmapPoint(x1, y1)
	x2, y2 = m.UnmapPoint(x2, y2)
	return clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
}

// Tensor is a prepared NCHW float32 tensor for one frame.
type Tensor struct {
	Data   []float32
	Shape  []int64
	Width  int
	Height int
	Meta   Meta
}

// Build renders one frame into the declared NCHW FP32 input tensor.
func Build(f *frame.Frame, spec model.IOSpec) (*Tensor, error) {
	if spec.Layout != "NCHW" {
		return nil, fmt.Errorf("unsupported input layout %q", spec.Layout)
	}
	if len(spec.Shape) != 4 || spec.Shape[0] != 1 || spec.Shape[1] != 3 {
		return nil, fmt.Errorf("expected NCHW [1,3,H,W] input shape, got %v", spec.Shape)
	}
	inH, inW := int(spec.Shape[2]), int(spec.Shape[3])
	if inH <= 0 || inW <= 0 {
		return nil, fmt.Errorf("input tensor height/width must be positive, got %dx%d", inH, inW)
	}

	src, err := toImage(f)
	if err != nil {
		return nil, err
	}
	bounds := src.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	resizeMode := "stretch"
	pp := spec.Preprocess
	var padR, padG, padB float32
	if pp != nil {
		if pp.Resize != "" {
			resizeMode = pp.Resize
		}
		padR, padG, padB = padRGB(pp)
	}

	var canvas *image.RGBA
	var meta Meta
	meta.SourceW, meta.SourceH, meta.InputW, meta.InputH = srcW, srcH, inW, inH

	switch resizeMode {
	case "stretch":
		canvas = resizeBilinear(src, inW, inH)
		meta.ScaleX = float32(inW) / float32(srcW)
		meta.ScaleY = float32(inH) / float32(srcH)
		meta.Scale = meta.ScaleX
	case "letterbox":
		scale := minF(float32(inW)/float32(srcW), float32(inH)/float32(srcH))
		rw, rh := maxI(1, int(float32(srcW)*scale+0.5)), maxI(1, int(float32(srcH)*scale+0.5))
		resized := resizeBilinear(src, rw, rh)
		canvas = image.NewRGBA(image.Rect(0, 0, inW, inH))
		fillRGB(canvas, padR, padG, padB)
		offX, offY := (inW-rw)/2, (inH-rh)/2
		copyRect(canvas, resized, offX, offY)
		meta.Scale = scale
		meta.ScaleX = scale
		meta.ScaleY = scale
		meta.PadX = float32(offX)
		meta.PadY = float32(offY)
	case "crop":
		// Fill the input aspect ratio, then take the centered crop.
		scale := maxF(float32(inW)/float32(srcW), float32(inH)/float32(srcH))
		rw, rh := maxI(inW, int(float32(srcW)*scale+0.5)), maxI(inH, int(float32(srcH)*scale+0.5))
		resized := resizeBilinear(src, rw, rh)
		canvas = image.NewRGBA(image.Rect(0, 0, inW, inH))
		offX, offY := (inW-rw)/2, (inH-rh)/2
		copyCenteredCrop(canvas, resized, offX, offY)
		meta.Scale = scale
		meta.ScaleX = scale
		meta.ScaleY = scale
		meta.PadX = float32(offX)
		meta.PadY = float32(offY)
	default:
		return nil, fmt.Errorf("unsupported resize mode %q", resizeMode)
	}

	scaleMul := float32(1)
	var mean, std []float32
	if pp != nil {
		var err error
		scaleMul, err = parseScale(pp.Scale)
		if err != nil {
			return nil, err
		}
		mean, std = pp.Mean, pp.Std
	}

	data := make([]float32, 3*inW*inH)
	plane := inW * inH
	for y := 0; y < inH; y++ {
		for x := 0; x < inW; x++ {
			r, g, b, _ := canvas.At(x, y).RGBA()
			// Normalization is descriptor-driven below: raw 0..255 values are
			// multiplied by Scale (e.g. 1/255), then mean/std are applied.
			rf, gf, bf := float32(r>>8), float32(g>>8), float32(b>>8)
			if spec.ColorOrder == "BGR" {
				rf, bf = bf, rf
			}
			idx := y*inW + x
			data[0*plane+idx] = normalize(rf, 0, mean, std, scaleMul)
			data[1*plane+idx] = normalize(gf, 1, mean, std, scaleMul)
			data[2*plane+idx] = normalize(bf, 2, mean, std, scaleMul)
		}
	}
	return &Tensor{Data: data, Shape: []int64{1, 3, int64(inH), int64(inW)}, Width: inW, Height: inH, Meta: meta}, nil
}

func toImage(f *frame.Frame) (image.Image, error) {
	switch f.Encoding {
	case frame.RGB24:
		if len(f.Bytes) < f.Width*f.Height*3 {
			return nil, fmt.Errorf("rgb24 frame too short: %d < %d", len(f.Bytes), f.Width*f.Height*3)
		}
		img := image.NewRGBA(image.Rect(0, 0, f.Width, f.Height))
		for i := 0; i < f.Width*f.Height; i++ {
			img.Pix[4*i] = f.Bytes[3*i]
			img.Pix[4*i+1] = f.Bytes[3*i+1]
			img.Pix[4*i+2] = f.Bytes[3*i+2]
			img.Pix[4*i+3] = 255
		}
		return img, nil
	case frame.JPEG:
		return jpeg.Decode(bytes.NewReader(f.Bytes))
	default:
		return nil, fmt.Errorf("unsupported frame encoding %q", f.Encoding)
	}
}

func padRGB(pp *model.PreprocessSpec) (float32, float32, float32) {
	v := pp.PadValue
	return v, v, v
}

// parseScale accepts forms like "1/255", "0.00392", "1", "2/255".
func parseScale(expr string) (float32, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return 1, nil
	}
	if parts := strings.Split(expr, "/"); len(parts) == 2 {
		a, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 32)
		b, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 32)
		if err1 == nil && err2 == nil && b != 0 {
			return float32(a / b), nil
		}
	}
	v, err := strconv.ParseFloat(expr, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid preprocess scale %q: %w", expr, err)
	}
	return float32(v), nil
}

func normalize(v float32, channel int, mean, std []float32, mul float32) float32 {
	v *= mul
	if len(mean) == 1 || len(mean) == 3 {
		if len(mean) == 1 {
			v -= mean[0]
		} else {
			v -= mean[channel]
		}
	}
	if len(std) == 1 || len(std) == 3 {
		var s float32
		if len(std) == 1 {
			s = std[0]
		} else {
			s = std[channel]
		}
		if s != 0 {
			v /= s
		}
	}
	return v
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func minF(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxF(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func maxI(a, b int) int {
	if a > b {
		return a
	}
	return b
}
