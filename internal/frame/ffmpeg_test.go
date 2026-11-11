// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func indexOf(args []string, want string) int {
	for i, a := range args {
		if a == want {
			return i
		}
	}
	return -1
}

func TestArgvSoftwarePipeline(t *testing.T) {
	s := &ffmpegSource{
		ref: ResolvedRef{Kind: "rtsp", URL: "rt://example.com/live", DecodeAccel: DecodeAccelSoftware},
		cfg: DefaultFFmpegConfig(),
	}
	args := s.argv(704, 576)
	if i := indexOf(args, "-hwaccel"); i >= 0 {
		t.Fatalf("software argv must not contain -hwaccel: %v", args)
	}
	vf := indexOf(args, "-vf")
	if vf < 0 || args[vf+1] != "scale=704:576" {
		t.Fatalf("software filter missing: %v", args)
	}
	iInput := indexOf(args, "-i")
	if iInput < 0 || iInput > vf {
		t.Fatalf("-i must precede -vf: %v", args)
	}
}

func TestArgvVAAPIPipeline(t *testing.T) {
	cfg := DefaultFFmpegConfig()
	s := &ffmpegSource{
		ref: ResolvedRef{
			Kind:        "rtsp",
			URL:         "rtsp://example.com/live",
			DecodeAccel: DecodeAccelVAAPI,
			VAAPIDevice: "/dev/dri/renderD129",
		},
		cfg: cfg,
	}
	args := s.argv(640, 480)

	iInit := indexOf(args, "-init_hw_device")
	if iInit < 0 || args[iInit+1] != "vaapi=va:/dev/dri/renderD129" {
		t.Fatalf("job-specific VA-API device init missing: %v", args)
	}
	iAccel := indexOf(args, "-hwaccel")
	iInput := indexOf(args, "-i")
	vf := indexOf(args, "-vf")
	if iAccel < 0 || iInput < 0 || vf < 0 || iInit >= iAccel || iAccel >= iInput || iInput >= vf {
		t.Fatalf("VA-API tokens must be ordered init < hwaccel < -i < -vf: %v", args)
	}
	if got := args[indexOf(args, "-hwaccel_output_format")+1]; got != "vaapi" {
		t.Fatalf("surfaces must stay on the device, got %q in %v", got, args)
	}
	if args[vf+1] != "scale_vaapi=640:480,hwdownload,format=nv12" {
		t.Fatalf("VA-API scaling/download filter missing: %v", args)
	}
}

func TestDecodeAccelValidation(t *testing.T) {
	m, err := NewManager(DefaultFFmpegConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.StopAll()

	if _, err := newSource(ResolvedRef{Kind: "jpeg", DecodeAccel: DecodeAccelVAAPI}, m.ffmpegCfg, nil); err == nil {
		t.Fatal("vaapi must be rejected for non-rtsp kinds")
	}
	if _, err := m.Subscribe(context.Background(), ResolvedRef{
		Kind: "jpeg", URL: "http://127.0.0.1:1/x", DecodeAccel: "cuda",
	}); err == nil {
		t.Fatal("unsupported DecodeAccel must be rejected")
	}
}

func TestConflictingDecodeAccelRejected(t *testing.T) {
	raw := mustJPEG(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(raw)
	}))
	defer srv.Close()

	m, err := NewManager(DefaultFFmpegConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.StopAll()

	ref := ResolvedRef{Kind: "jpeg", URL: srv.URL, DecodeAccel: DecodeAccelSoftware}
	sub, err := m.Subscribe(context.Background(), ref)
	if err != nil {
		t.Fatal(err)
	}
	defer sub.Release()
	if _, err := m.Subscribe(context.Background(), ResolvedRef{
		Kind: "jpeg", URL: srv.URL, DecodeAccel: DecodeAccelVAAPI,
	}); err == nil {
		t.Fatal("a shared URL must reject conflicting DecodeAccel settings")
	}
}

func TestDeviceNodeGIDMissingNode(t *testing.T) {
	if _, ok := deviceNodeGID("/dev/dri/does-not-exist-xyz"); ok {
		t.Fatal("missing device node must report no GID")
	}
}
