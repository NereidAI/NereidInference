// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package preprocess

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/model"
)

// BuildROI crops a normalized source-frame box and then applies the consumer
// descriptor's ordinary resize, color and normalization rules. The returned
// tensor always has static batch 1; per-face batching is intentionally absent.
func BuildROI(f *frame.Frame, spec model.IOSpec, box [4]float32) (*Tensor, error) {
	src, err := toImage(f)
	if err != nil {
		return nil, err
	}
	crop, err := cropNormalized(src, box)
	if err != nil {
		return nil, err
	}
	return Build(frameFromRGBA(crop), spec)
}

// BuildFivePoint aligns five normalized source landmarks to the destination
// template declared by the consumer model, producing its exact [1,3,H,W]
// input. The similarity warp subsumes ROI extraction and resize; box is
// validated and documents the detection ROI associated with the landmarks.
func BuildFivePoint(f *frame.Frame, spec model.IOSpec, box [4]float32, points [][2]float32) (*Tensor, error) {
	if spec.Preprocess == nil || spec.Preprocess.Alignment == nil {
		return nil, fmt.Errorf("five-point alignment requires io.inputs[0].preprocess.alignment.template")
	}
	if len(points) != 5 || len(spec.Preprocess.Alignment.Template) != 5 {
		return nil, fmt.Errorf("five-point alignment requires exactly five source and template points")
	}
	if len(spec.Shape) != 4 || spec.Shape[0] != 1 || spec.Shape[1] != 3 || spec.Shape[2] <= 0 || spec.Shape[3] <= 0 {
		return nil, fmt.Errorf("five-point alignment expects static NCHW [1,3,H,W], got %v", spec.Shape)
	}
	src, err := toImage(f)
	if err != nil {
		return nil, err
	}
	if _, err := cropNormalized(src, box); err != nil {
		return nil, err
	}
	bounds := src.Bounds()
	source := make([][2]float64, 5)
	for i, point := range points {
		if !finitePoint(point[0], point[1]) {
			return nil, fmt.Errorf("source landmark %d is not finite", i)
		}
		source[i] = [2]float64{
			float64(bounds.Min.X) + float64(point[0])*float64(bounds.Dx()),
			float64(bounds.Min.Y) + float64(point[1])*float64(bounds.Dy()),
		}
	}
	template := make([][2]float64, 5)
	for i, point := range spec.Preprocess.Alignment.Template {
		if !finitePoint(point[0], point[1]) {
			return nil, fmt.Errorf("template landmark %d is not finite", i)
		}
		template[i] = [2]float64{float64(point[0]), float64(point[1])}
	}
	forward, err := estimateSimilarity(source, template)
	if err != nil {
		return nil, err
	}
	inverse, err := invertAffine(forward)
	if err != nil {
		return nil, err
	}
	inW, inH := int(spec.Shape[3]), int(spec.Shape[2])
	canvas := image.NewRGBA(image.Rect(0, 0, inW, inH))
	padR, padG, padB := padRGB(spec.Preprocess)
	for y := 0; y < inH; y++ {
		for x := 0; x < inW; x++ {
			sx, sy := applyAffine(inverse, float64(x), float64(y))
			canvas.SetRGBA(x, y, sampleBilinear(src, sx, sy, color.RGBA{R: uint8(padR), G: uint8(padG), B: uint8(padB), A: 255}))
		}
	}
	// Geometry is already exact. A 1:1 stretch through Build reuses the single
	// descriptor-driven color/scale/mean/std implementation without another
	// resize or a second interpretation of normalization.
	buildSpec := spec
	pp := *spec.Preprocess
	pp.Resize = "stretch"
	pp.Alignment = nil
	buildSpec.Preprocess = &pp
	return Build(frameFromRGBA(canvas), buildSpec)
}

type affine struct {
	a, b, c float64
	d, e, f float64
}

// estimateSimilarity implements the same five-point Umeyama fit as OpenCV
// FaceRecognizerSF: rotation or reflection, one uniform scale and translation.
// The closed-form 2-D polar decomposition avoids a matrix dependency while
// rejecting rank-deficient geometry instead of inventing shear or axis scales.
func estimateSimilarity(source, destination [][2]float64) (affine, error) {
	if len(source) != 5 || len(destination) != 5 {
		return affine{}, fmt.Errorf("similarity estimation requires five point pairs")
	}
	var srcMean, dstMean [2]float64
	for i := range source {
		for axis := 0; axis < 2; axis++ {
			srcMean[axis] += source[i][axis] / 5
			dstMean[axis] += destination[i][axis] / 5
		}
	}
	// covariance is destination^T * source, divided by five exactly as in
	// OpenCV. The divisor cancels from the final scale but is retained here to
	// make the reference algorithm and its tolerances explicit.
	var covariance [2][2]float64
	var sourceVariance float64
	for i := range source {
		sx := source[i][0] - srcMean[0]
		sy := source[i][1] - srcMean[1]
		dx := destination[i][0] - dstMean[0]
		dy := destination[i][1] - dstMean[1]
		covariance[0][0] += dx * sx / 5
		covariance[0][1] += dx * sy / 5
		covariance[1][0] += dy * sx / 5
		covariance[1][1] += dy * sy / 5
		sourceVariance += (sx*sx + sy*sy) / 5
	}
	if sourceVariance <= 1e-12 {
		return affine{}, fmt.Errorf("degenerate five-point similarity source: zero variance")
	}
	a00, a01 := covariance[0][0], covariance[0][1]
	a10, a11 := covariance[1][0], covariance[1][1]
	determinant := a00*a11 - a01*a10
	determinantScale := math.Abs(a00*a11) + math.Abs(a01*a10)
	if math.Abs(determinant) <= 1e-12*math.Max(1, determinantScale) {
		return affine{}, fmt.Errorf("degenerate five-point similarity source: rank-deficient covariance")
	}

	// For det(A)>=0, R=[c -s;s c]. For det(A)<0, OpenCV's
	// D=diag(1,-1) yields the closest reflection Q=[c s;s -c].
	// In both cases the vector norm is the signed singular-value sum used
	// by OpenCV's scale=(s0*d0+s1*d1)/sourceVariance.
	var c, s float64
	reflected := determinant < 0
	if reflected {
		c, s = a00-a11, a01+a10
	} else {
		c, s = a00+a11, a10-a01
	}
	norm := math.Hypot(c, s)
	if norm <= 1e-12 {
		return affine{}, fmt.Errorf("degenerate five-point similarity source: zero rotation norm")
	}
	c, s = c/norm, s/norm
	scale := norm / sourceVariance
	var rotation affine
	if reflected {
		rotation = affine{a: c, b: s, d: s, e: -c}
	} else {
		rotation = affine{a: c, b: -s, d: s, e: c}
	}
	return affine{
		a: scale * rotation.a,
		b: scale * rotation.b,
		c: dstMean[0] - scale*(rotation.a*srcMean[0]+rotation.b*srcMean[1]),
		d: scale * rotation.d,
		e: scale * rotation.e,
		f: dstMean[1] - scale*(rotation.d*srcMean[0]+rotation.e*srcMean[1]),
	}, nil
}

func invertAffine(in affine) (affine, error) {
	det := in.a*in.e - in.b*in.d
	scale := math.Abs(in.a*in.e) + math.Abs(in.b*in.d)
	if math.Abs(det) < 1e-12*math.Max(1, scale) {
		return affine{}, fmt.Errorf("five-point similarity transform is not invertible")
	}
	return affine{
		a: in.e / det, b: -in.b / det, c: (in.b*in.f - in.e*in.c) / det,
		d: -in.d / det, e: in.a / det, f: (in.d*in.c - in.a*in.f) / det,
	}, nil
}

func applyAffine(m affine, x, y float64) (float64, float64) {
	return m.a*x + m.b*y + m.c, m.d*x + m.e*y + m.f
}

func cropNormalized(src image.Image, box [4]float32) (*image.RGBA, error) {
	if !finitePoint(box[0], box[1]) || !finitePoint(box[2], box[3]) {
		return nil, fmt.Errorf("ROI coordinates must be finite")
	}
	if box[0] < 0 || box[1] < 0 || box[2] > 1 || box[3] > 1 || box[2] <= box[0] || box[3] <= box[1] {
		return nil, fmt.Errorf("ROI must be ordered and inside normalized source bounds, got %v", box)
	}
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	x1 := bounds.Min.X + int(math.Floor(float64(box[0])*float64(w)))
	y1 := bounds.Min.Y + int(math.Floor(float64(box[1])*float64(h)))
	x2 := bounds.Min.X + int(math.Ceil(float64(box[2])*float64(w)))
	y2 := bounds.Min.Y + int(math.Ceil(float64(box[3])*float64(h)))
	if x1 < bounds.Min.X {
		x1 = bounds.Min.X
	}
	if y1 < bounds.Min.Y {
		y1 = bounds.Min.Y
	}
	if x2 > bounds.Max.X {
		x2 = bounds.Max.X
	}
	if y2 > bounds.Max.Y {
		y2 = bounds.Max.Y
	}
	if x2 <= x1 || y2 <= y1 {
		return nil, fmt.Errorf("ROI collapses to an empty pixel rectangle")
	}
	out := image.NewRGBA(image.Rect(0, 0, x2-x1, y2-y1))
	for y := 0; y < out.Bounds().Dy(); y++ {
		for x := 0; x < out.Bounds().Dx(); x++ {
			out.Set(x, y, src.At(x1+x, y1+y))
		}
	}
	return out, nil
}

func sampleBilinear(src image.Image, x, y float64, pad color.RGBA) color.RGBA {
	bounds := src.Bounds()
	if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) ||
		x < float64(bounds.Min.X) || y < float64(bounds.Min.Y) || x > float64(bounds.Max.X-1) || y > float64(bounds.Max.Y-1) {
		return pad
	}
	x0, y0 := int(math.Floor(x)), int(math.Floor(y))
	x1, y1 := x0+1, y0+1
	if x1 >= bounds.Max.X {
		x1 = bounds.Max.X - 1
	}
	if y1 >= bounds.Max.Y {
		y1 = bounds.Max.Y - 1
	}
	fx, fy := x-float64(x0), y-float64(y0)
	convert := func(c color.Color) color.RGBA { return color.RGBAModel.Convert(c).(color.RGBA) }
	c00, c10 := convert(src.At(x0, y0)), convert(src.At(x1, y0))
	c01, c11 := convert(src.At(x0, y1)), convert(src.At(x1, y1))
	mix := func(v00, v10, v01, v11 uint8) uint8 {
		top := float64(v00)*(1-fx) + float64(v10)*fx
		bottom := float64(v01)*(1-fx) + float64(v11)*fx
		return uint8(math.Round(top*(1-fy) + bottom*fy))
	}
	return color.RGBA{
		R: mix(c00.R, c10.R, c01.R, c11.R),
		G: mix(c00.G, c10.G, c01.G, c11.G),
		B: mix(c00.B, c10.B, c01.B, c11.B),
		A: 255,
	}
}

func frameFromRGBA(img *image.RGBA) *frame.Frame {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	data := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pixel := img.RGBAAt(img.Bounds().Min.X+x, img.Bounds().Min.Y+y)
			idx := 3 * (y*w + x)
			data[idx], data[idx+1], data[idx+2] = pixel.R, pixel.G, pixel.B
		}
	}
	return &frame.Frame{Encoding: frame.RGB24, Width: w, Height: h, Bytes: data}
}

func finitePoint(values ...float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}
