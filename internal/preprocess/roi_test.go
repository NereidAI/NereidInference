// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package preprocess

import (
	"math"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/model"
)

func TestBuildROICropsBeforeDescriptorResizeAndNormalization(t *testing.T) {
	source := &frame.Frame{Encoding: frame.RGB24, Width: 4, Height: 2, Bytes: []byte{
		255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
		255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
	}}
	spec := model.IOSpec{
		Name: "face", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW", ColorOrder: "RGB",
		Preprocess: &model.PreprocessSpec{Resize: "stretch", Scale: "1/255"},
	}
	tensor, err := BuildROI(source, spec, [4]float32{0.5, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := tensor.Shape; len(got) != 4 || got[0] != 1 || got[2] != 2 || got[3] != 2 {
		t.Fatalf("shape = %v", got)
	}
	// Left output column is blue; right output column is white.
	for _, idx := range []int{0, 2} {
		if tensor.Data[idx] != 0 || tensor.Data[idx+4] != 0 || tensor.Data[idx+8] != 1 {
			t.Fatalf("blue output pixel %d has wrong normalized RGB: %v", idx, tensor.Data)
		}
	}
	for _, idx := range []int{1, 3} {
		if tensor.Data[idx] != 1 || tensor.Data[idx+4] != 1 || tensor.Data[idx+8] != 1 {
			t.Fatalf("white output pixel %d has wrong normalized RGB: %v", idx, tensor.Data)
		}
	}
}

func TestBuildFivePointIdentityMatchesWholeImageAndUsesDescriptor(t *testing.T) {
	source := &frame.Frame{Encoding: frame.RGB24, Width: 2, Height: 2, Bytes: []byte{
		255, 0, 0, 0, 255, 0,
		0, 0, 255, 255, 255, 255,
	}}
	spec := model.IOSpec{
		Name: "face", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW", ColorOrder: "BGR",
		Preprocess: &model.PreprocessSpec{
			Resize: "stretch", Scale: "1/255",
			Alignment: &model.AlignmentSpec{Template: [][2]float32{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}},
		},
	}
	points := [][2]float32{{0, 0}, {0.5, 0}, {0, 0.5}, {0.5, 0.5}, {0.25, 0.25}}
	aligned, err := BuildFivePoint(source, spec, [4]float32{0, 0, 1, 1}, points)
	if err != nil {
		t.Fatal(err)
	}
	plainSpec := spec
	pp := *spec.Preprocess
	pp.Alignment = nil
	plainSpec.Preprocess = &pp
	plain, err := Build(source, plainSpec)
	if err != nil {
		t.Fatal(err)
	}
	for i := range plain.Data {
		if math.Abs(float64(plain.Data[i]-aligned.Data[i])) > 0.01 {
			t.Fatalf("identity similarity tensor[%d] = %g, want %g", i, aligned.Data[i], plain.Data[i])
		}
	}
	if aligned.Shape[0] != 1 || aligned.Data[0] != 0 || aligned.Data[8] != 1 {
		t.Fatalf("descriptor BGR order was not applied: shape=%v data=%v", aligned.Shape, aligned.Data)
	}
}

func TestEstimateSimilarityMatchesOpenCVNoisyReference(t *testing.T) {
	// Reference generated with OpenCV FaceRecognizerSF's 2x2 SVD/Umeyama
	// algorithm. Source points include shear and non-uniform scaling, which a
	// general affine fit would reproduce but a similarity transform must not.
	source := [][2]float64{
		{58.390125, 73.37573}, {86.55065, 69.59864}, {75.58065, 97.65506},
		{67.094265, 125.92022}, {90.414535, 122.79234},
	}
	destination := [][2]float64{
		{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366},
		{41.5493, 92.3655}, {70.7299, 92.2041},
	}
	got, err := estimateSimilarity(source, destination)
	if err != nil {
		t.Fatal(err)
	}
	want := affine{
		a: 0.8504688242255837, b: -0.09992893183311444, c: 1.5054602768615553,
		d: 0.09992893183311441, e: 0.8504688242255838, f: -18.888472692877855,
	}
	for name, pair := range map[string][2]float64{
		"a": {got.a, want.a}, "b": {got.b, want.b}, "c": {got.c, want.c},
		"d": {got.d, want.d}, "e": {got.e, want.e}, "f": {got.f, want.f},
	} {
		if math.Abs(pair[0]-pair[1]) > 1e-9 {
			t.Errorf("%s = %.15g, want %.15g", name, pair[0], pair[1])
		}
	}
	column1 := math.Hypot(got.a, got.d)
	column2 := math.Hypot(got.b, got.e)
	if math.Abs(column1-column2) > 1e-12 || math.Abs(got.a*got.b+got.d*got.e) > 1e-12 {
		t.Fatalf("transform contains shear or non-uniform scale: %+v", got)
	}
}

func TestEstimateSimilarityRecoversRotationAndReflection(t *testing.T) {
	destination := [][2]float64{{0, 0}, {4, 0}, {2, 3}, {1, 5}, {3, 5}}
	for _, want := range []affine{
		{a: 0, b: -2, c: 10, d: 2, e: 0, f: 20},
		{a: -2, b: 0, c: 10, d: 0, e: 2, f: 20},
	} {
		inverse, err := invertAffine(want)
		if err != nil {
			t.Fatal(err)
		}
		source := make([][2]float64, len(destination))
		for i, point := range destination {
			source[i][0], source[i][1] = applyAffine(inverse, point[0], point[1])
		}
		got, err := estimateSimilarity(source, destination)
		if err != nil {
			t.Fatal(err)
		}
		for i, point := range source {
			u, v := applyAffine(got, point[0], point[1])
			if math.Hypot(u-destination[i][0], v-destination[i][1]) > 1e-9 {
				t.Fatalf("similarity %+v maps point %d to (%g,%g), want %v", got, i, u, v, destination[i])
			}
		}
	}
}

func TestBuildFivePointRejectsDegenerateOrInvalidGeometry(t *testing.T) {
	source := &frame.Frame{Encoding: frame.RGB24, Width: 2, Height: 2, Bytes: make([]byte, 12)}
	spec := model.IOSpec{
		Name: "face", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW",
		Preprocess: &model.PreprocessSpec{Alignment: &model.AlignmentSpec{Template: [][2]float32{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}}},
	}
	if _, err := BuildFivePoint(source, spec, [4]float32{0, 0, 1, 1}, make([][2]float32, 5)); err == nil {
		t.Fatal("collinear landmarks must fail similarity estimation")
	}
	if _, err := BuildFivePoint(source, spec, [4]float32{0, 0, 1, 1}, [][2]float32{{0, 0}}); err == nil {
		t.Fatal("missing landmarks must fail")
	}
	if _, err := BuildROI(source, spec, [4]float32{0.8, 0, 0.2, 1}); err == nil {
		t.Fatal("inverted ROI must fail")
	}
}
