// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package publish

import (
	"bytes"
	"image"
	"image/jpeg"
	"testing"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
)

func TestHubRemoveClosesSubscribers(t *testing.T) {
	h := NewHub()
	sub := h.subscribe("camera-01")
	h.Remove("camera-01")

	select {
	case _, ok := <-sub.ch:
		if ok {
			t.Fatal("expected subscriber channel to be closed")
		}
	case <-time.After(time.Second):
		t.Fatal("subscriber was not closed by Remove")
	}
	// A client goroutine may call unsubscribe while Remove is also closing.
	h.unsubscribe("camera-01", sub)
}

func TestHubDropOldestAndSnapshot(t *testing.T) {
	h := NewHub()
	first := &Frame{}
	second := &Frame{}
	sub := h.subscribe("camera-02")
	defer h.unsubscribe("camera-02", sub)

	h.Update("camera-02", first)
	h.Update("camera-02", second)
	got := <-sub.ch
	if got != second {
		t.Fatal("subscriber did not receive newest frame")
	}
	if snap, ok := h.snapshot("camera-02"); !ok || snap != second {
		t.Fatalf("snapshot = %v, %v", snap, ok)
	}
}

func TestEncodeJPEGIncludesDetectionOverlay(t *testing.T) {
	canvas := image.NewRGBA(image.Rect(0, 0, 40, 30))
	for i := range canvas.Pix {
		canvas.Pix[i] = 255
	}
	var src bytes.Buffer
	if err := jpeg.Encode(&src, canvas, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatal(err)
	}
	result := &decode.DetectionResult{Detections: []decode.Detection{{
		Class: 0,
		Label: "mask",
		Score: 0.95,
		Box:   [4]float32{0.1, 0.2, 0.9, 0.8},
	}}}
	b, err := EncodeJPEG(&Frame{
		Source:   &frame.Frame{Encoding: frame.JPEG, Width: 40, Height: 30, Bytes: src.Bytes()},
		Result:   result,
		Annotate: true,
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	greenPixels := 0
	for y := 5; y < 25; y++ {
		r, g, _, _ := img.At(4, y).RGBA()
		if r < 120<<8 && g > 150<<8 {
			greenPixels++
		}
	}
	if greenPixels == 0 {
		t.Fatal("encoded evidence JPEG does not contain the green detection box")
	}
}

func TestSplitTail(t *testing.T) {
	tests := []struct {
		name       string
		tail       string
		wantDevice string
		wantKind   string
	}{
		{name: "mjpeg", tail: "inference-yolov8n-jpeg-01.mjpeg", wantDevice: "inference-yolov8n-jpeg-01", wantKind: "mjpeg"},
		{name: "jpeg", tail: "inference-yolov8n-jpeg-01/frame.jpeg", wantDevice: "inference-yolov8n-jpeg-01", wantKind: "jpeg"},
		{name: "unknown", tail: "inference-yolov8n-jpeg-01.txt", wantDevice: "", wantKind: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDevice, gotKind := splitTail(tt.tail)
			if gotDevice != tt.wantDevice || gotKind != tt.wantKind {
				t.Fatalf("splitTail(%q) = (%q, %q), want (%q, %q)", tt.tail, gotDevice, gotKind, tt.wantDevice, tt.wantKind)
			}
		})
	}
}
