// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package preprocess

import (
	"testing"

	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/model"
)

func TestBuildStretchRGBTensor(t *testing.T) {
	src := &frame.Frame{Encoding: frame.RGB24, Width: 2, Height: 2, Bytes: []byte{
		255, 0, 0, 0, 255, 0,
		0, 0, 255, 255, 255, 255,
	}}
	spec := model.IOSpec{
		Name:       "images",
		Shape:      []int64{1, 3, 2, 2},
		Layout:     "NCHW",
		ColorOrder: "RGB",
		Preprocess: &model.PreprocessSpec{Resize: "stretch", Scale: "1/255"},
	}
	got, err := Build(src, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 12 {
		t.Fatalf("tensor length = %d, want 12", len(got.Data))
	}
	const eps = 0.0001
	plane := 4
	checks := map[int]float32{
		0:           1, // R at (0,0), 255 * (1/255)
		plane + 1:   1, // G at (1,0)
		2*plane + 2: 1, // B at (0,1)
		3:           1, // white pixel R
		plane + 3:   1, // white pixel G
		2*plane + 3: 1, // white pixel B
	}
	for idx, want := range checks {
		if got.Data[idx] < want-eps || got.Data[idx] > want+eps {
			t.Fatalf("data[%d] = %v, want %v; tensor=%v", idx, got.Data[idx], want, got.Data)
		}
	}
}

func TestBuildCenterCropMeta(t *testing.T) {
	src := &frame.Frame{Encoding: frame.RGB24, Width: 1, Height: 2, Bytes: make([]byte, 6)}
	spec := model.IOSpec{
		Name: "images", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW",
		Preprocess: &model.PreprocessSpec{Resize: "crop", Scale: "1"},
	}
	got, err := Build(src, spec)
	if err != nil {
		t.Fatal(err)
	}
	if got.Meta.Scale != 2 || got.Meta.PadX != 0 || got.Meta.PadY != -1 {
		t.Fatalf("unexpected crop meta: %+v", got.Meta)
	}
	_, y1, _, y2 := got.Meta.UnmapBox(0, 0, 1, 1)
	if y1 != 0.25 || y2 != 0.75 {
		t.Fatalf("vertical crop unmap = %v..%v, want .25..75", y1, y2)
	}
}

func TestBuildLetterboxMeta(t *testing.T) {
	src := &frame.Frame{Encoding: frame.RGB24, Width: 2, Height: 1, Bytes: make([]byte, 6)}
	spec := model.IOSpec{
		Name: "images", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW",
		Preprocess: &model.PreprocessSpec{Resize: "letterbox", Scale: "1", PadValue: 114},
	}
	got, err := Build(src, spec)
	if err != nil {
		t.Fatal(err)
	}
	if got.Meta.Scale != 1 || got.Meta.PadX != 0 || got.Meta.PadY != 0 {
		t.Fatalf("unexpected letterbox meta: %+v", got.Meta)
	}
	x1, y1, x2, y2 := got.Meta.UnmapBox(0, 0, 1, 1)
	if x1 != 0 || x2 != 1 || y1 != 0 || y2 != 1 {
		t.Fatalf("unmap = %v,%v,%v,%v, want edges", x1, y1, x2, y2)
	}
}
