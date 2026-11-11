// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"
)

func mustJPEG(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 75}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func waitFrame(t *testing.T, sub *Subscription) {
	t.Helper()
	select {
	case f := <-sub.C:
		if f.Encoding != JPEG || f.Width != 2 || f.Height != 1 {
			t.Fatalf("unexpected frame: %+v", f)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for fan-out frame")
	}
}

func waitClosed(t *testing.T, sub *Subscription) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case _, ok := <-sub.C:
			if !ok {
				return
			}
		case <-deadline:
			t.Fatal("subscription channel was not closed after release")
		}
	}
}

func TestManagerFanOutAndRelease(t *testing.T) {
	raw := mustJPEG(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(raw)
	}))
	defer srv.Close()

	m, err := NewManager(DefaultFFmpegConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.StopAll()

	ctx := context.Background()
	ref := ResolvedRef{Kind: "jpeg", URL: srv.URL}
	sub1, err := m.Subscribe(ctx, ref)
	if err != nil {
		t.Fatal(err)
	}
	sub2, err := m.Subscribe(ctx, ref)
	if err != nil {
		t.Fatal(err)
	}
	waitFrame(t, sub1)
	waitFrame(t, sub2)

	sub1.Release()
	sub1.Release() // repeated release must be safe
	waitClosed(t, sub1)

	select {
	case <-sub2.C:
	default:
	}
	sub2.Release()
	waitClosed(t, sub2)
}

func TestSubscribeRejectsConflictingRef(t *testing.T) {
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

	jpegSub, err := m.Subscribe(context.Background(), ResolvedRef{Kind: "jpeg", URL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer jpegSub.Release()
	if _, err := m.Subscribe(context.Background(), ResolvedRef{Kind: "mpjpeg", URL: srv.URL}); err == nil {
		t.Fatal("expected conflicting source kind to be rejected")
	}
}

func TestRenderTemplateKeepsURLAsSingleArg(t *testing.T) {
	cfg := DefaultFFmpegConfig()
	url := "rtsp://user:a b@host/path?x=-evil"
	args := renderTemplate("-rtsp_transport tcp -timeout {timeout_us} -i {url}", url, cfg)
	var found int
	for _, arg := range args {
		if arg == url {
			found++
		}
		if arg == "-evil" {
			t.Fatal("URL text was tokenized into an ffmpeg argument")
		}
	}
	if found != 1 {
		t.Fatalf("URL appeared %d times in args %v", found, args)
	}
}

func TestValidateExtraArgs(t *testing.T) {
	// FFmpegExtra tokens are rendered after -i (output side), so -r is an
	// output frame-rate resample; the cloud low-latency combo is accepted
	// here but its input-side latency effect belongs to the input template.
	for _, extra := range []string{
		"-r 30", "-re", "-threads=2", "-r 2",
		"-fflags nobuffer -flags low_delay",
	} {
		if err := validateExtraArgs(extra); err != nil {
			t.Fatalf("expected %q to be allowed: %v", extra, err)
		}
	}
	for _, extra := range []string{"-i /etc/passwd", "-y", "http://evil"} {
		if err := validateExtraArgs(extra); err == nil {
			t.Fatalf("expected %q to be rejected", extra)
		}
	}
}

func TestNewManagerValidatesConfig(t *testing.T) {
	cfg := DefaultFFmpegConfig()
	cfg.ExtraArgs = "-i /etc/passwd"
	if _, err := NewManager(cfg, nil); err == nil {
		t.Fatal("expected invalid global extra args to be rejected")
	}
	cfg = DefaultFFmpegConfig()
	cfg.InputTemplates["rtsp"] = "-rtsp_transport tcp"
	if _, err := NewManager(cfg, nil); err == nil {
		t.Fatal("expected template without {url} to be rejected")
	}
}

func TestValidateGeometry(t *testing.T) {
	if err := validateGeometry(7680, 4320); err != nil {
		t.Fatal(err)
	}
	for _, dims := range [][2]int{{0, 100}, {100, 0}, {-1, 100}, {7681, 4320}, {100000, 100000}} {
		if err := validateGeometry(dims[0], dims[1]); err == nil {
			t.Fatalf("geometry %v was accepted", dims)
		}
	}
}

func TestONVIFResponseIsCapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(bytes.Repeat([]byte("x"), maxONVIFJSON+1))
	}))
	defer srv.Close()

	f := NewFederation(srv.URL, srv.URL, nil, FederationConfig{})
	if _, _, err := f.getJSON(context.Background(), srv.URL); err == nil {
		t.Fatal("expected oversized ONVIF response to be rejected")
	}
}

func TestEdgeXSnapshotSource(t *testing.T) {
	jpegBytes := mustJPEG(t)
	payload := map[string]any{
		"event": map[string]any{
			"readings": []map[string]any{{
				"resourceName": "Snapshot",
				"valueType":    "Binary",
				"mediaType":    "image/jpeg",
				"binaryValue":  base64.StdEncoding.EncodeToString(jpegBytes),
			}},
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(payload)
	}))
	defer srv.Close()

	src := &edgexSnapshotSource{url: srv.URL, client: srv.Client()}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	frames, err := src.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case f := <-frames:
		if f.Width != 2 || f.Height != 1 || f.Encoding != JPEG {
			t.Fatalf("unexpected frame %+v", f)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for snapshot frame")
	}
}

func TestFederationV3RoutesAndSecretHint(t *testing.T) {
	const metadata = `{"device":{"protocols":{"Onvif":{"MACAddress":"aa:bb:cc:dd:ee:ff"}}}}`
	const command = `{"event":{"readings":[{"resourceName":"StreamUri","objectValue":{"MediaUri":{"Uri":"rtsp://camera/stream"}}}]}}`
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v3/device/name/camera-01", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v3/device/name/camera-01" {
			t.Fatalf("unexpected metadata path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(metadata))
	})
	mux.HandleFunc("/api/v3/device/name/camera-01/StreamUri", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v3/device/name/camera-01/StreamUri" {
			t.Fatalf("unexpected command path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(command))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	f := NewFederation(srv.URL, srv.URL, func(name string) (map[string]string, error) {
		if name != "camera-auth" {
			t.Fatalf("secret name = %q, want camera-auth", name)
		}
		return map[string]string{"username": "viewer", "password": "password"}, nil
	}, FederationConfig{})
	ref, err := f.Resolve(context.Background(), "camera-01", "camera-auth", "")
	if err != nil {
		t.Fatal(err)
	}
	u, err := url.Parse(ref.URL)
	if err != nil {
		t.Fatal(err)
	}
	if user := u.User.Username(); user != "viewer" {
		t.Fatalf("username = %q, want viewer", user)
	}
	if pass, ok := u.User.Password(); !ok || pass != "password" {
		t.Fatalf("password present=%v", ok)
	}
}

func TestFederationInvalidateRefreshesURI(t *testing.T) {
	var metadataHits atomic.Int32
	const metadata = `{"device":{"protocols":{"Onvif":{"SecretName":""}}}}`
	const command = `{"event":{"readings":[{"resourceName":"StreamUri","value":"rtsp://camera/stream"}]}}`
	mux := http.NewServeMux()
	mux.HandleFunc("/metadata/", func(w http.ResponseWriter, _ *http.Request) {
		metadataHits.Add(1)
		_, _ = w.Write([]byte(metadata))
	})
	mux.HandleFunc("/command/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(command))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	f := NewFederation(srv.URL+"/metadata", srv.URL+"/command", nil, FederationConfig{})
	ref, err := f.Resolve(context.Background(), "camera-01", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if ref.URL != "rtsp://camera/stream" {
		t.Fatalf("unexpected ref %+v", ref)
	}
	if _, err := f.Resolve(context.Background(), "camera-01", "", ""); err != nil {
		t.Fatal(err)
	}
	if got := metadataHits.Load(); got != 1 {
		t.Fatalf("metadata hits after cached resolve = %d, want 1", got)
	}
	f.Invalidate("camera-01")
	if _, err := f.Resolve(context.Background(), "camera-01", "", ""); err != nil {
		t.Fatal(err)
	}
	if got := metadataHits.Load(); got != 2 {
		t.Fatalf("metadata hits after invalidate = %d, want 2", got)
	}
}

// TestFederationProfileSelection verifies a non-empty ProfileToken issues a
// parameterized PUT command and that main/sub profiles resolve and cache
// independently; Invalidate drops every profile of the camera.
// TestFederationProfileSelection verifies the device-onvif-camera 4.x command
// shape: a non-empty ProfileToken is a GET StreamUri whose jsonObject query is
// base64(JSON{ProfileToken, StreamSetup}), the resource is read-only (PUT 405)
// and main/sub profiles resolve and cache independently; Invalidate drops every
// profile of a camera.
func TestFederationProfileSelection(t *testing.T) {
	const metadata = `{"device":{"protocols":{"Onvif":{"SecretName":""}}}}`
	var getProfile1, getProfile2, defaultGet, legacyGet atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v3/device/name/camera-01", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(metadata))
	})
	mux.HandleFunc("/api/v3/device/name/camera-01/StreamUri", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			// The StreamUri device resource is read-only.
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		encoded := r.URL.Query().Get("jsonObject")
		if encoded == "" {
			defaultGet.Add(1)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Errorf("jsonObject is not base64: %v", err)
		}
		var req streamSetupRequest
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatal(err)
		}
		if req.StreamSetup.Stream != "RTP-Unicast" ||
			req.StreamSetup.Transport.Protocol != "RTSP" {
			t.Fatalf("unexpected StreamSetup: %+v", req.StreamSetup)
		}
		switch req.ProfileToken {
		case "Profile_1":
			getProfile1.Add(1)
		case "Profile_2":
			getProfile2.Add(1)
		default:
			t.Errorf("unexpected ProfileToken %q", req.ProfileToken)
		}
		resp := map[string]any{"event": map[string]any{"readings": []map[string]any{{
			"resourceName": "StreamUri",
			"objectValue": map[string]any{
				"MediaUri": map[string]any{"Uri": "rtsp://camera/stream?profile=" + req.ProfileToken},
			},
		}}}}
		_ = json.NewEncoder(w).Encode(resp)
	})
	mux.HandleFunc("/api/v3/device/name/camera-01/GetStreamUri", func(w http.ResponseWriter, _ *http.Request) {
		legacyGet.Add(1)
		w.WriteHeader(http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	f := NewFederation(srv.URL, srv.URL, nil, FederationConfig{})
	ctx := context.Background()

	main, err := f.Resolve(ctx, "camera-01", "", "Profile_1")
	if err != nil {
		t.Fatal(err)
	}
	sub, err := f.Resolve(ctx, "camera-01", "", "Profile_2")
	if err != nil {
		t.Fatal(err)
	}
	if main.URL == sub.URL {
		t.Fatalf("main/sub resolved to the same URI %q", main.URL)
	}
	if main.URL != "rtsp://camera/stream?profile=Profile_1" ||
		sub.URL != "rtsp://camera/stream?profile=Profile_2" {
		t.Fatalf("unexpected URIs main=%s sub=%s", main.URL, sub.URL)
	}

	// Both profiles are cached independently: a repeat resolve sends no GET.
	if _, err := f.Resolve(ctx, "camera-01", "", "Profile_1"); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Resolve(ctx, "camera-01", "", "Profile_2"); err != nil {
		t.Fatal(err)
	}
	if getProfile1.Load() != 1 || getProfile2.Load() != 1 {
		t.Fatalf("cached profiles issued new GETs: p1=%d p2=%d", getProfile1.Load(), getProfile2.Load())
	}

	// Invalidate drops every profile variant of the camera.
	f.Invalidate("camera-01")
	if _, err := f.Resolve(ctx, "camera-01", "", "Profile_1"); err != nil {
		t.Fatal(err)
	}
	if getProfile1.Load() != 2 {
		t.Fatalf("invalidate did not drop the profile cache: p1=%d", getProfile1.Load())
	}

	// An empty token uses the parameter-less default-profile path; the
	// stand-in rejects it, proving the two code paths stay distinct.
	if _, err := f.Resolve(ctx, "camera-01", "", ""); err == nil {
		t.Fatal("default GET must fail when the stand-in requires jsonObject")
	}
	if defaultGet.Load() != 1 {
		t.Fatalf("default-profile StreamUri GET count = %d, want 1", defaultGet.Load())
	}
	// Profile-scoped requests succeed on the first command and must never
	// fall through to the legacy spelling; the parameter-less default path
	// does fall through, proving why.
	if legacyGet.Load() != 1 {
		t.Fatalf("GetStreamUri fallback count = %d, want 1 (default path only)", legacyGet.Load())
	}
}
