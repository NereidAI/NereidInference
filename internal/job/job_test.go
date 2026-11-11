// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/infer"
	pb "github.com/edgexfoundry/device-inference/internal/infer/kserve/grpcv2"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"google.golang.org/grpc"
)

type testLogger struct{ t *testing.T }

func (l testLogger) Infof(format string, args ...any)  { l.t.Logf("INFO "+format, args...) }
func (l testLogger) Errorf(format string, args ...any) { l.t.Logf("ERROR "+format, args...) }
func (l testLogger) Debugf(format string, args ...any) { l.t.Logf("DEBUG "+format, args...) }

func mustNewManager(t *testing.T) *frame.Manager {
	t.Helper()
	m, err := frame.NewManager(frame.DefaultFFmpegConfig(), nil)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestFromPropertiesDefaults(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":      "m",
		"SourceDevice": "camera-01",
		"FPS":          "2.5",
		"Trigger":      "periodic",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SourceKind != "onvif-camera" || cfg.FPS != 2.5 || cfg.Trigger != TriggerPeriodic {
		t.Fatalf("unexpected config: %+v", cfg)
	}
}

func TestFromPropertiesNormalizesBindingTargetDevice(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":      "m",
		"SourceDevice": "camera-01",
		"TargetDevice": "gpu.0",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TargetDevice != "GPU.0" {
		t.Fatalf("TargetDevice = %q, want GPU.0", cfg.TargetDevice)
	}
}

func TestFromPropertiesRejectsUnsafeTargetDevice(t *testing.T) {
	_, err := FromProperties(map[string]any{
		"ModelID":      "m",
		"SourceDevice": "camera-01",
		"TargetDevice": "GPU; rm -rf /",
	})
	if err == nil {
		t.Fatal("expected unsafe target device to be rejected")
	}
}

func TestFromPropertiesStreamProfile(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":       "m",
		"SourceDevice":  "camera-01",
		"StreamProfile": " Profile_2 ",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.StreamProfile != "Profile_2" {
		t.Fatalf("StreamProfile=%q, want trimmed Profile_2", cfg.StreamProfile)
	}

	if _, err := FromProperties(map[string]any{
		"ModelID":       "m",
		"SourceDevice":  "camera-01",
		"StreamProfile": "Profile 2",
	}); err == nil {
		t.Fatal("StreamProfile with a space must be rejected")
	}
	if _, err := FromProperties(map[string]any{
		"ModelID":       "m",
		"SourceKind":    "onvif-snapshot",
		"SourceDevice":  "camera-01",
		"StreamProfile": "Profile_2",
	}); err == nil {
		t.Fatal("StreamProfile on a non-federated source must be rejected")
	}
}

func TestFromPropertiesONVIFSnapshot(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":      "m",
		"SourceKind":   "onvif-snapshot",
		"SourceDevice": "camera-01",
		"Trigger":      "continuous",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SourceURL != "" {
		t.Fatalf("snapshot federation must not require a direct URL: %+v", cfg)
	}
}

func TestFromPropertiesDecodeAccel(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":     "m",
		"SourceKind":  "rtsp",
		"SourceURL":   "rtsp://camera/live",
		"DecodeAccel": " VAAPI ",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DecodeAccel != frame.DecodeAccelVAAPI {
		t.Fatalf("DecodeAccel = %q, want vaapi", cfg.DecodeAccel)
	}
	cfg, err = FromProperties(map[string]any{
		"ModelID": "m", "SourceKind": "rtsp", "SourceURL": "rtsp://camera/live",
		"DecodeAccel": "vaapi", "VAAPIDevice": "renderD129",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VAAPIDevice != "/dev/dri/renderD129" {
		t.Fatalf("VAAPIDevice = %q", cfg.VAAPIDevice)
	}
	if federated, err := FromProperties(map[string]any{
		"ModelID": "m", "SourceKind": "onvif-camera", "SourceDevice": "camera-01",
		"DecodeAccel": "vaapi",
	}); err != nil || federated.VAAPIDevice != "" {
		t.Fatalf("federated onvif-camera vaapi must be allowed: %+v %v", federated, err)
	}
	if got, err := FromProperties(map[string]any{
		"ModelID": "m", "SourceKind": "rtsp", "SourceURL": "rtsp://camera/live",
	}); err != nil || got.DecodeAccel != frame.DecodeAccelSoftware {
		t.Fatalf("default DecodeAccel = %q, %v", got.DecodeAccel, err)
	}
	for name, props := range map[string]map[string]any{
		"bad-value": {
			"ModelID": "m", "SourceKind": "rtsp", "SourceURL": "rtsp://camera/live",
			"DecodeAccel": "qsv",
		},
		"vaapi-on-snapshot": {
			"ModelID": "m", "SourceKind": "onvif-snapshot", "SourceDevice": "camera-01",
			"DecodeAccel": "vaapi",
		},
		"vaapi-on-jpeg": {
			"ModelID": "m", "SourceKind": "jpeg", "SourceURL": "http://camera/snap",
			"DecodeAccel": "vaapi",
		},
		"bad-node": {
			"ModelID": "m", "SourceKind": "rtsp", "SourceURL": "rtsp://camera/live",
			"DecodeAccel": "vaapi", "VAAPIDevice": "../dri/renderD129",
		},
		"device-without-accel": {
			"ModelID": "m", "SourceKind": "rtsp", "SourceURL": "rtsp://camera/live",
			"VAAPIDevice": "renderD129",
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := FromProperties(props); err == nil {
				t.Fatal("expected DecodeAccel validation error")
			}
		})
	}
}

func TestFromPropertiesRejectsInvalidValues(t *testing.T) {
	base := func() map[string]any {
		return map[string]any{"ModelID": "m", "SourceURL": "http://camera/snap", "SourceKind": "jpeg"}
	}
	for name, mutate := range map[string]func(map[string]any){
		"fps-nan":      func(p map[string]any) { p["FPS"] = "NaN" },
		"fps-too-high": func(p map[string]any) { p["FPS"] = "61" },
		"threshold":    func(p map[string]any) { p["ConfThreshold"] = "1.5" },
		"sampling":     func(p map[string]any) { p["Sampling"] = "{invalid" },
		"ffmpeg":       func(p map[string]any) { p["FFmpegExtra"] = "-i /etc/passwd" },
		"period": func(p map[string]any) {
			p["Trigger"] = "periodic"
			p["PeriodMs"] = "10000000000000"
		},
		"idle": func(p map[string]any) { p["MaxIdleMs"] = "10000000000000" },
	} {
		t.Run(name, func(t *testing.T) {
			props := base()
			mutate(props)
			if _, err := FromProperties(props); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestResolveONVIFSnapshotReference(t *testing.T) {
	j := &Job{
		cfg:  Config{SourceKind: "onvif-snapshot", SourceDevice: "camera/1"},
		deps: Deps{CoreCommandURL: "http://core-command:59882/"},
	}
	ref, err := j.resolveRef()
	if err != nil {
		t.Fatal(err)
	}
	if ref.Kind != "onvif-snapshot" {
		t.Fatalf("kind = %q", ref.Kind)
	}
	want := "http://core-command:59882/api/v3/device/name/camera%2F1/Snapshot"
	if ref.URL != want {
		t.Fatalf("URL = %q, want %q", ref.URL, want)
	}
}

func TestResolveFederatedStreamCarriesExtraArgs(t *testing.T) {
	const metadata = `{"device":{"protocols":{"Onvif":{"SecretName":""}}}}`
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v3/device/name/camera-1", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(metadata))
	})
	// device-onvif-camera 4.x: profile selection is GET StreamUri with a
	// base64(JSON{ProfileToken,StreamSetup}) jsonObject query parameter.
	mux.HandleFunc("/api/v3/device/name/camera-1/StreamUri", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		raw, err := base64.StdEncoding.DecodeString(r.URL.Query().Get("jsonObject"))
		if err != nil {
			t.Fatalf("jsonObject is not base64: %v", err)
		}
		var req struct {
			ProfileToken string `json:"ProfileToken"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatal(err)
		}
		if req.ProfileToken != "Profile_2" {
			t.Errorf("ProfileToken=%q, want Profile_2", req.ProfileToken)
		}
		resp := map[string]any{"event": map[string]any{"readings": []map[string]any{{
			"objectValue": map[string]any{
				"MediaUri": map[string]any{"Uri": "rtsp://camera/sub"},
			},
		}}}}
		_ = json.NewEncoder(w).Encode(resp)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	fed := frame.NewFederation(srv.URL, srv.URL, nil, frame.FederationConfig{})
	j := &Job{
		cfg: Config{
			SourceKind: "onvif-camera", SourceDevice: "camera-1",
			StreamProfile: "Profile_2",
			FFmpegExtra:   "-fflags nobuffer -flags low_delay",
		},
		deps: Deps{Federation: fed},
		ctx:  context.Background(),
	}
	ref, err := j.resolveRef()
	if err != nil {
		t.Fatal(err)
	}
	if ref.ExtraArgs != "-fflags nobuffer -flags low_delay" {
		t.Fatalf("ExtraArgs=%q, want the cloud low-latency combo", ref.ExtraArgs)
	}
	if ref.URL != "rtsp://camera/sub" {
		t.Fatalf("URL=%q", ref.URL)
	}
}

func TestScheduleIntervalUsesPeriodicInterval(t *testing.T) {
	j := &Job{cfg: Config{Trigger: TriggerPeriodic, PeriodMs: 60000}}
	if got := j.scheduleInterval(); got != time.Minute {
		t.Fatalf("periodic interval = %s, want 1m", got)
	}
	j = &Job{cfg: Config{Trigger: TriggerContinuous, FPS: 1}}
	if got := j.scheduleInterval(); got != time.Second {
		t.Fatalf("continuous interval = %s, want 1s", got)
	}
}

func TestOpenAITextJobCycle(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"chat-model"}]}`))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	inbox := t.TempDir()
	writeLLMPackage(t, filepath.Join(inbox, "llm.tar.gz"))
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	j := New("llm-device", Config{
		ModelID: "test-chat", ModelVersion: "1.0.0",
		SourceKind: "none", Trigger: TriggerCommand,
		Endpoint: server.URL, MaxInferMs: 5000,
	}, Deps{
		Catalog:  catalog,
		Backends: infer.NewRegistry("", nil),
		Decoders: decode.NewRegistry(),
		Log:      testLogger{t: t},
		JSONCap:  1 << 20,
	})
	j.Start()
	defer j.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	readings, err := j.Invoke(ctx, "ping")
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for _, r := range readings {
		if r.Name == "Text" {
			text, _ = r.Value.(string)
		}
	}
	if text != "hello" {
		t.Fatalf("text = %q, readings=%v", text, readings)
	}
}

func TestOVMSVisionJobUsesBindingTargetServable(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	fake := &fakeManagedKServeServer{t: t}
	pb.RegisterGRPCInferenceServiceServer(grpcServer, fake)
	go func() { _ = grpcServer.Serve(lis) }()
	defer grpcServer.Stop()

	reloadServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer reloadServer.Close()

	var jpegBuf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	_ = jpeg.Encode(&jpegBuf, img, &jpeg.Options{Quality: 90})
	camera := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(jpegBuf.Bytes())
	}))
	defer camera.Close()

	root := t.TempDir()
	inbox := t.TempDir()
	writeManagedVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"))
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	serverManager, err := modelserver.New(modelserver.Config{
		ModelRoot:      root,
		ContainerRoot:  "/models",
		ReloadEndpoint: reloadServer.URL,
		GRPCEndpoint:   lis.Addr().String(),
		Catalog:        catalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	serverManager.Start()
	defer serverManager.Stop()
	servables := modelserver.NewRegistry(serverManager)

	j := New("managed-vision-device", Config{
		ModelID: "test-detect", ModelVersion: "1.0.0", TargetDevice: "GPU",
		SourceKind: string(frame.JPEG), SourceURL: camera.URL,
		Trigger: TriggerCommand, Endpoint: lis.Addr().String(),
		MaxInferMs: 5000,
	}, Deps{
		Catalog:        catalog,
		Frames:         mustNewManager(t),
		Servables:      servables,
		Backends:       infer.NewRegistry("", nil),
		Decoders:       decode.NewRegistry(),
		Log:            testLogger{t: t},
		JSONCap:        1 << 20,
		FrameTTLFactor: 2,
	})
	j.Start()
	defer j.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := j.Invoke(ctx, ""); err != nil {
		t.Fatal(err)
	}
	want := modelserver.ServedName("test-detect", "1.0.0", "GPU")
	if fake.modelName != want {
		t.Fatalf("inference model = %q, want binding servable %q", fake.modelName, want)
	}
}

func TestTritonVisionJobUsesNFPackageLease(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	fake := &fakeManagedKServeServer{t: t}
	pb.RegisterGRPCInferenceServiceServer(grpcServer, fake)
	go func() { _ = grpcServer.Serve(lis) }()
	defer grpcServer.Stop()

	wantName := modelserver.ServedName("test-detect", "1.0.0", "GPU")
	control := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v2/repository/models/"+wantName+"/load":
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodGet && r.URL.Path == "/v2/models/"+wantName+"/ready":
			w.WriteHeader(http.StatusOK)
		case r.Method == http.MethodPost && r.URL.Path == "/v2/repository/models/"+wantName+"/unload":
			w.WriteHeader(http.StatusOK)
		default:
			http.Error(w, "unexpected fake Triton request", http.StatusNotFound)
		}
	}))
	defer control.Close()

	var jpegBuf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	_ = jpeg.Encode(&jpegBuf, img, &jpeg.Options{Quality: 90})
	camera := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(jpegBuf.Bytes())
	}))
	defer camera.Close()

	root, inbox := t.TempDir(), t.TempDir()
	writeVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"), modelserver.FamilyTriton)
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	manager, err := modelserver.NewTriton(modelserver.TritonConfig{
		ModelRoot: root, ContainerRoot: "/models", ControlEndpoint: control.URL,
		GRPCEndpoint: lis.Addr().String(), Catalog: catalog,
		LoadTimeout: 5 * time.Second, ReadyTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()

	j := New("triton-nf-vision-device", Config{
		ModelID: "test-detect", ModelVersion: "1.0.0", TargetDevice: "GPU",
		SourceKind: string(frame.JPEG), SourceURL: camera.URL,
		Trigger: TriggerCommand, MaxInferMs: 5000,
	}, Deps{
		Catalog: catalog, Frames: mustNewManager(t), Servables: modelserver.NewRegistry(manager),
		Backends: infer.NewRegistry("", nil), Decoders: decode.NewRegistry(),
		Log: testLogger{t: t}, JSONCap: 1 << 20, FrameTTLFactor: 2,
	})
	j.Start()
	defer j.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := j.Invoke(ctx, ""); err != nil {
		t.Fatal(err)
	}
	if fake.modelName != wantName {
		t.Fatalf("inference model = %q, want Triton lease servable %q", fake.modelName, wantName)
	}
}

func TestManagedVisionJobRequiresFamilyManager(t *testing.T) {
	for _, family := range []string{modelserver.FamilyOVMS, modelserver.FamilyTriton} {
		t.Run(family, func(t *testing.T) {
			root, inbox := t.TempDir(), t.TempDir()
			if family == modelserver.FamilyOVMS {
				writeManagedVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"))
			} else {
				writeVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"), family)
			}
			catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
			if err != nil {
				t.Fatal(err)
			}
			if err := catalog.Start(); err != nil {
				t.Fatal(err)
			}
			defer catalog.Stop()
			backends := infer.NewRegistry("", nil)
			defer backends.Close()
			j := New("missing-manager-"+family, Config{
				ModelID: "test-detect", ModelVersion: "1.0.0", SourceKind: string(frame.JPEG),
				SourceURL: "http://127.0.0.1:1/unused.jpg", Trigger: TriggerCommand,
				Endpoint: "127.0.0.1:1",
			}, Deps{
				Catalog: catalog, Frames: mustNewManager(t), Backends: backends,
				Decoders: decode.NewRegistry(), Servables: modelserver.NewRegistry(),
				Log: testLogger{t: t}, JSONCap: 1 << 20, FrameTTLFactor: 2,
			})
			j.Start()
			defer j.Stop()
			deadline := time.Now().Add(2 * time.Second)
			for time.Now().Before(deadline) {
				if j.StatsSnapshot()["LastError"] == ReasonConfigInvalid {
					return
				}
				time.Sleep(10 * time.Millisecond)
			}
			t.Fatalf("managed %s model bypassed its missing manager: %+v", family, j.StatsSnapshot())
		})
	}
}

func TestKServeVisionJobCycle(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(grpcServer, &fakeKServeServer{t: t})
	go func() { _ = grpcServer.Serve(lis) }()
	defer grpcServer.Stop()

	var jpegBuf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	_ = jpeg.Encode(&jpegBuf, img, &jpeg.Options{Quality: 90})
	camera := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(jpegBuf.Bytes())
	}))
	defer camera.Close()

	root := t.TempDir()
	inbox := t.TempDir()
	writeVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"), "external")
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	j := New("vision-device", Config{
		ModelID: "test-detect", ModelVersion: "1.0.0",
		SourceKind: string(frame.JPEG), SourceURL: camera.URL,
		Trigger: TriggerCommand, Endpoint: lis.Addr().String(),
		MaxInferMs: 5000,
	}, Deps{
		Catalog:        catalog,
		Frames:         mustNewManager(t),
		Backends:       infer.NewRegistry("", nil),
		Decoders:       decode.NewRegistry(),
		Log:            testLogger{t: t},
		JSONCap:        1 << 20,
		FrameTTLFactor: 2,
	})
	j.Start()
	defer j.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	readings, err := j.Invoke(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	var count int32
	var payload string
	for _, r := range readings {
		switch r.Name {
		case "DetectionCount":
			count, _ = r.Value.(int32)
		case "DetectionJSON":
			payload, _ = r.Value.(string)
		}
	}
	if count != 1 || payload == "" {
		t.Fatalf("unexpected readings count=%d json=%q values=%v", count, payload, readings)
	}
}

func TestContinuousDeadSourceEscalatesToFailed(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	deadURL := "http://" + lis.Addr().String()
	_ = lis.Close() // every poll fails with connection refused; no frame ever arrives

	lis, err = net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(grpcServer, &fakeKServeServer{t: t})
	go func() { _ = grpcServer.Serve(lis) }()
	defer grpcServer.Stop()

	root := t.TempDir()
	inbox := t.TempDir()
	writeVisionPackage(t, filepath.Join(inbox, "vision.tar.gz"), "external")
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	// threshold 1 makes the first failed cycle DEGRADED; the hard FAIL state
	// still needs failedEscalationFactor consecutive failures. Each frame-less
	// cycle blocks on nextFrame's 2s production floor, so this end-to-end
	// assertion takes ~12s by design.
	const threshold = 1
	j := New("dead-source-device", Config{
		ModelID: "test-detect", ModelVersion: "1.0.0",
		SourceKind: string(frame.JPEG), SourceURL: deadURL,
		Trigger: TriggerContinuous, FPS: 20,
		ErrorThreshold: threshold, MaxIdleMs: 3_600_000,
		Endpoint: lis.Addr().String(), MaxInferMs: 5000,
	}, Deps{
		Catalog:        catalog,
		Frames:         mustNewManager(t),
		Backends:       infer.NewRegistry("", nil),
		Decoders:       decode.NewRegistry(),
		Log:            testLogger{t: t},
		JSONCap:        1 << 20,
		FrameTTLFactor: 2,
	})
	j.Start()
	defer j.Stop()

	deadline := time.Now().Add(40 * time.Second)
	var sawSourceError bool
	for time.Now().Before(deadline) {
		snap := j.StatsSnapshot()
		if snap["LastError"] == ReasonSourceNoFrame {
			sawSourceError = true
		}
		// Only FAILED escalations reached through source errors count;
		// model/config force-failures use different LastError constants.
		if sawSourceError && snap["State"] == StateFailed {
			if snap["LastError"] != ReasonSourceNoFrame {
				t.Fatalf("FAILED with reason=%v, want %s", snap["LastError"], ReasonSourceNoFrame)
			}
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	snap := j.StatsSnapshot()
	t.Fatalf("dead continuous source never FAILED: state=%v reason=%v errors=%v",
		snap["State"], snap["LastError"], snap["ErrorTotal"])
}

type fakeManagedKServeServer struct {
	pb.UnimplementedGRPCInferenceServiceServer
	t         *testing.T
	modelName string
}

func (s *fakeManagedKServeServer) ModelReady(_ context.Context, req *pb.ModelReadyRequest) (*pb.ModelReadyResponse, error) {
	s.modelName = req.GetName()
	if req.GetName() != modelserver.ServedName("test-detect", "1.0.0", "GPU") {
		return &pb.ModelReadyResponse{Ready: false}, nil
	}
	return &pb.ModelReadyResponse{Ready: true}, nil
}

func (s *fakeManagedKServeServer) ModelInfer(_ context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	s.modelName = req.GetModelName()
	return &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name: "output", Datatype: "FP32", Shape: []int64{1, 6, 1},
			Contents: &pb.InferTensorContents{Fp32Contents: []float32{5, 5, 2, 2, 0.1, 0.9}},
		}},
	}, nil
}

type fakeKServeServer struct {
	pb.UnimplementedGRPCInferenceServiceServer
	t *testing.T
}

func (s *fakeKServeServer) ModelReady(context.Context, *pb.ModelReadyRequest) (*pb.ModelReadyResponse, error) {
	return &pb.ModelReadyResponse{Ready: true}, nil
}

func (s *fakeKServeServer) ModelInfer(_ context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	if req.ModelName != "detect-model" {
		s.t.Fatalf("model name = %q", req.ModelName)
	}
	// [1, 4 box values + 2 classes, 1 anchor], centered on 10x10 input.
	return &pb.ModelInferResponse{
		Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name: "output", Datatype: "FP32", Shape: []int64{1, 6, 1},
			Contents: &pb.InferTensorContents{Fp32Contents: []float32{
				5, 5, 2, 2, 0.1, 0.9,
			}},
		}},
	}, nil
}

func writeManagedVisionPackage(t *testing.T, path string) {
	t.Helper()
	xml := []byte("<net></net>")
	bin := []byte("weights")
	labels := []byte("background\nobject\n")
	xmlSum := sha256.Sum256(xml)
	binSum := sha256.Sum256(bin)
	labelsSum := sha256.Sum256(labels)
	manifest := []byte(fmt.Sprintf(`{
      "schema": "nereid-model/1",
      "model": {"id": "test-detect", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
      "artifact": {"format": "openvino-ir", "precision": "FP16", "files": [
        {"path":"labels.txt","bytes":%d,"sha256":"%s"},
        {"path":"repository/detect-model/1/model.xml","bytes":%d,"sha256":"%s"},
        {"path":"repository/detect-model/1/model.bin","bytes":%d,"sha256":"%s"}
      ]},
      "runtime": {
        "backend": "kserve",
        "model_name": "detect-model",
        "repository_path": "repository/detect-model",
        "server": {"family": "ovms", "min_version": "2026.2.1"}
      },
      "io": {
        "inputs": [{"name": "images", "shape": [1,3,10,10], "layout": "NCHW", "dtype": "FP32",
          "preprocess": {"resize": "stretch", "scale": "1/255", "color_order": "RGB"}}],
        "outputs": [{"name": "output", "shape": [1,6,1], "layout": "NCA", "dtype": "FP32"}]
      },
      "decoder": {"kind": "detection.anchors", "output": "output", "threshold": 0.25}
    }`,
		len(labels), hex.EncodeToString(labelsSum[:]),
		len(xml), hex.EncodeToString(xmlSum[:]),
		len(bin), hex.EncodeToString(binSum[:])))
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	writeTarEntry := func(name string, b []byte) {
		if err := tw.WriteHeader(&tar.Header{Name: name, Size: int64(len(b)), Mode: 0o644, Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(b); err != nil {
			t.Fatal(err)
		}
	}
	writeTarEntry("nereid-model.json", manifest)
	writeTarEntry("labels.txt", labels)
	writeTarEntry("repository/detect-model/1/model.xml", xml)
	writeTarEntry("repository/detect-model/1/model.bin", bin)
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeLLMPackage(t *testing.T, path string) {
	t.Helper()
	manifest := []byte(`{
      "schema": "nereid-model/1",
      "model": {"id": "test-chat", "version": "1.0.0", "task": "chat"},
      "runtime": {"backend": "openai", "model_name": "chat-model"},
      "io": {"inputs": [], "outputs": []},
      "decoder": {"kind": "passthrough.text"}
    }`)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(&tar.Header{Name: "nereid-model.json", Size: int64(len(manifest)), Mode: 0o644, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(manifest); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeVisionPackage(t *testing.T, path, family string) {
	t.Helper()
	labels := []byte("background\nobject\n")
	onnx := []byte("fake-onnx-model")
	labelsSum := sha256.Sum256(labels)
	onnxSum := sha256.Sum256(onnx)
	manifest := []byte(fmt.Sprintf(`{
      "schema": "nereid-model/1",
      "model": {"id": "test-detect", "version": "1.0.0", "task": "detection", "labels_file": "labels.txt"},
      "artifact": {"format": "onnx", "precision": "FP32", "files": [
        {"path": "labels.txt", "bytes": %d, "sha256": "%s"},
        {"path": "repository/detect-model/1/model.onnx", "bytes": %d, "sha256": "%s"}
      ]},
      "runtime": {"backend": "kserve", "model_name": "detect-model",
        "server": {"family": "%s"}},
      "io": {
        "inputs": [{"name": "images", "shape": [1,3,10,10], "layout": "NCHW", "dtype": "FP32",
          "preprocess": {"resize": "stretch", "scale": "1/255", "color_order": "RGB"}}],
        "outputs": [{"name": "output", "shape": [1,6,1], "layout": "NCA", "dtype": "FP32"}]
      },
      "decoder": {"kind": "detection.anchors", "output": "output", "threshold": 0.25}
    }`, len(labels), hex.EncodeToString(labelsSum[:]),
		len(onnx), hex.EncodeToString(onnxSum[:]), family))
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	writeTarEntry := func(name string, b []byte) {
		if err := tw.WriteHeader(&tar.Header{Name: name, Size: int64(len(b)), Mode: 0o644, Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(b); err != nil {
			t.Fatal(err)
		}
	}
	writeTarEntry("nereid-model.json", manifest)
	writeTarEntry("labels.txt", labels)
	writeTarEntry("repository/detect-model/1/model.onnx", onnx)
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
}
