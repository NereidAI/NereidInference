// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/infer"
	pb "github.com/edgexfoundry/device-inference/internal/infer/kserve/grpcv2"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/pipeline/testsupport"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type recordedEmission struct {
	device   string
	readings []decode.Reading
	origin   int64
}

type recordingEmitter struct {
	mu     sync.Mutex
	events []recordedEmission
}

func (e *recordingEmitter) Emit(_ context.Context, device string, readings []decode.Reading, origin int64) {
	e.mu.Lock()
	e.events = append(e.events, recordedEmission{device: device, readings: append([]decode.Reading(nil), readings...), origin: origin})
	e.mu.Unlock()
}

func (e *recordingEmitter) snapshot() []recordedEmission {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]recordedEmission(nil), e.events...)
}

type fakeFacePipelineKServe struct {
	pb.UnimplementedGRPCInferenceServiceServer

	mu             sync.Mutex
	detectorName   string
	embeddingName  string
	score          float32
	detectorCalls  int
	embeddingCalls int
	embeddingShape []int64
	notReady       map[string]bool
	readyCalls     map[string]int
}

func (s *fakeFacePipelineKServe) ModelReady(_ context.Context, req *pb.ModelReadyRequest) (*pb.ModelReadyResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.readyCalls == nil {
		s.readyCalls = map[string]int{}
	}
	s.readyCalls[req.GetName()]++
	known := req.GetName() == s.detectorName || req.GetName() == s.embeddingName
	return &pb.ModelReadyResponse{Ready: known && !s.notReady[req.GetName()]}, nil
}

func (s *fakeFacePipelineKServe) ModelInfer(_ context.Context, req *pb.ModelInferRequest) (*pb.ModelInferResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch req.GetModelName() {
	case s.detectorName:
		s.detectorCalls++
		if len(req.GetInputs()) != 1 || !equalInt64s(req.GetInputs()[0].GetShape(), []int64{1, 3, 64, 64}) {
			return nil, status.Errorf(codes.InvalidArgument, "YuNet input shape = %v", req.GetInputs())
		}
		return facePipelineYuNetResponse(s.score), nil
	case s.embeddingName:
		s.embeddingCalls++
		if len(req.GetInputs()) != 1 || req.GetInputs()[0].GetName() != "sface-input" {
			return nil, status.Errorf(codes.InvalidArgument, "unexpected SFace inputs: %v", req.GetInputs())
		}
		s.embeddingShape = append([]int64(nil), req.GetInputs()[0].GetShape()...)
		if !equalInt64s(s.embeddingShape, []int64{1, 3, 112, 112}) {
			return nil, status.Errorf(codes.InvalidArgument, "SFace input shape = %v", s.embeddingShape)
		}
		vector := make([]float32, 128)
		for i := range vector {
			vector[i] = float32(i+1) / 128
		}
		return &pb.ModelInferResponse{Outputs: []*pb.ModelInferResponse_InferOutputTensor{{
			Name: "embedding", Datatype: "FP32", Shape: []int64{1, 128},
			Contents: &pb.InferTensorContents{Fp32Contents: vector},
		}}}, nil
	default:
		return nil, status.Errorf(codes.NotFound, "unexpected model %q", req.GetModelName())
	}
}

func (s *fakeFacePipelineKServe) setScore(score float32) {
	s.mu.Lock()
	s.score = score
	s.mu.Unlock()
}

func (s *fakeFacePipelineKServe) counts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.detectorCalls, s.embeddingCalls
}

func (s *fakeFacePipelineKServe) setReady(name string, ready bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.notReady == nil {
		s.notReady = map[string]bool{}
	}
	s.notReady[name] = !ready
}

func (s *fakeFacePipelineKServe) readyCallCounts() (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readyCalls[s.detectorName], s.readyCalls[s.embeddingName]
}

func TestPipelineCommandJPEGToKServeAndEdgeX(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	fake := &fakeFacePipelineKServe{
		detectorName:  modelserver.ServedName("yunet", "1.0.0", "CPU"),
		embeddingName: modelserver.ServedName("sface", "1.0.0", "CPU"),
		score:         0.9,
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(grpcServer, fake)
	go func() { _ = grpcServer.Serve(listener) }()
	defer grpcServer.Stop()

	reload := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer reload.Close()

	var encodedJPEG bytes.Buffer
	if err := jpeg.Encode(&encodedJPEG, image.NewRGBA(image.Rect(0, 0, 64, 64)), &jpeg.Options{Quality: 90}); err != nil {
		t.Fatal(err)
	}
	camera := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(encodedJPEG.Bytes())
	}))
	defer camera.Close()

	root, inbox := t.TempDir(), t.TempDir()
	yunetPackage := filepath.Join(inbox, "yunet.tar.gz")
	writeFacePipelinePackage(t, yunetPackage, "yunet")
	writeFacePipelinePackage(t, filepath.Join(inbox, "sface.tar.gz"), "sface")
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()

	manager, err := modelserver.New(modelserver.Config{
		ModelRoot: root, ContainerRoot: "/models", ReloadEndpoint: reload.URL,
		GRPCEndpoint: listener.Addr().String(), Catalog: catalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()
	pipelines, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	// The platform ships no compiled-in graph; tests seed the sample
	// face-recognition definition through the public Replace channel.
	if err := pipelines.Replace([]pipeline.Definition{testsupport.FaceRecognitionDefinition()}); err != nil {
		t.Fatal(err)
	}
	backends := infer.NewRegistry("", nil)
	defer backends.Close()
	emitter := &recordingEmitter{}
	job := New("face-command", Config{
		PipelineID: testsupport.FaceRecognitionID,
		SourceKind: "jpeg", SourceURL: camera.URL, Trigger: TriggerCommand,
		MaxInferMs: 5000, PublishMode: PublishAll,
	}, Deps{
		Catalog: catalog, Frames: mustNewManager(t), Backends: backends, Decoders: decode.NewRegistry(),
		Servables: modelserver.NewRegistry(manager), Pipelines: pipelines, Log: testLogger{t: t},
		Emitter: emitter, JSONCap: 1 << 20, FrameTTLFactor: 2,
	})
	job.Start()
	defer job.Stop()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	readings, err := job.Invoke(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	first, raw := decodePipelineReading(t, readings)
	if first.Schema != "nereid.pipeline/1" || first.PipelineID != testsupport.FaceRecognitionID || first.FrameID != 1 {
		t.Fatalf("unexpected first PipelineJSON envelope: %s", raw)
	}
	if len(first.Objects) != 1 || first.Objects[0].ID != 1 || first.Objects[0].ParentID != 0 || first.Objects[0].Type != "face" {
		t.Fatalf("unexpected typed objects: %s", raw)
	}
	if len(first.Objects[0].Keypoints) != 5 || len(first.Objects[0].Embedding) != 128 {
		t.Fatalf("missing five-point/128-D products: %s", raw)
	}
	if strings.Contains(raw, "jpeg") || strings.Contains(raw, "tensor") || strings.Contains(raw, "Float32") {
		t.Fatalf("raw source or tensors leaked into PipelineJSON: %s", raw)
	}
	if detectorCalls, embeddingCalls := fake.counts(); detectorCalls != 1 || embeddingCalls != 1 {
		t.Fatalf("gRPC calls = detector %d, SFace %d; want 1/1", detectorCalls, embeddingCalls)
	}
	assertEmitterPipelinePayload(t, emitter.snapshot(), 0, raw)
	refs := job.PipelineModelRefs()
	if len(refs) != 2 || refs[0].ID != "yunet" || refs[1].ID != "sface" {
		t.Fatalf("resolved model binding refs = %+v", refs)
	}

	// A healthy empty frame still emits objects:[] on PublishMode=all and does
	// not call SFace again. This is the same command path used for enrollment.
	fake.setScore(0)
	readings, err = job.Invoke(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	second, raw := decodePipelineReading(t, readings)
	if second.FrameID != 2 || second.Objects == nil || len(second.Objects) != 0 || !strings.Contains(raw, `"objects":[]`) {
		t.Fatalf("healthy empty command event = %s", raw)
	}
	if detectorCalls, embeddingCalls := fake.counts(); detectorCalls != 2 || embeddingCalls != 1 {
		t.Fatalf("empty-frame gRPC calls = detector %d, SFace %d; want 2/1", detectorCalls, embeddingCalls)
	}
	assertEmitterPipelinePayload(t, emitter.snapshot(), 1, raw)

	// Register a newest version whose models are absent. Resolution fails while
	// the v1 graph and its two exact package references remain active.
	v1, _ := pipelines.Lookup(testsupport.FaceRecognitionID)
	v2 := testsupport.FaceRecognitionDefinition()
	v2.Version = "2.0.0"
	v2.Nodes[1].Model.ModelID = "missing-yunet"
	v2.Nodes[3].Model.ModelID = "missing-sface"
	if err := pipelines.Replace([]pipeline.Definition{v1, v2}); err != nil {
		t.Fatal(err)
	}
	waitForPipelineCondition(t, 10*time.Second, func() bool {
		return job.StatsSnapshot()["State"] == StateFailed
	}, "failed v2 resolution")
	if got := job.PipelineModelRefs(); len(got) != 2 {
		t.Fatalf("failed upgrade discarded the still-active v1 graph: %+v", got)
	}

	// Removing the old detector must reset v1 even though newest-v2 no longer
	// references that package. The prior implementation ignored this event.
	if err := os.Remove(yunetPackage); err != nil {
		t.Fatal(err)
	}
	waitForPipelineCondition(t, 15*time.Second, func() bool {
		return len(job.PipelineModelRefs()) == 0
	}, "old active graph reset after package removal")
}

func TestPipelineActiveUpgradeRetriesReadinessWithoutCatalogEvent(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	fake := &fakeFacePipelineKServe{
		detectorName:  modelserver.ServedName("yunet", "1.0.0", "CPU"),
		embeddingName: modelserver.ServedName("sface", "1.0.0", "CPU"),
		score:         0.9,
	}
	grpcServer := grpc.NewServer()
	pb.RegisterGRPCInferenceServiceServer(grpcServer, fake)
	go func() { _ = grpcServer.Serve(listener) }()
	defer grpcServer.Stop()

	reload := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer reload.Close()
	var encodedJPEG bytes.Buffer
	if err := jpeg.Encode(&encodedJPEG, image.NewRGBA(image.Rect(0, 0, 64, 64)), &jpeg.Options{Quality: 90}); err != nil {
		t.Fatal(err)
	}
	camera := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		_, _ = w.Write(encodedJPEG.Bytes())
	}))
	defer camera.Close()

	root, inbox := t.TempDir(), t.TempDir()
	writeFacePipelinePackage(t, filepath.Join(inbox, "yunet.tar.gz"), "yunet")
	writeFacePipelinePackage(t, filepath.Join(inbox, "sface.tar.gz"), "sface")
	catalog, err := modelcatalog.New(modelcatalog.Config{Root: root, InboxEnabled: true, InboxDir: inbox})
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Start(); err != nil {
		t.Fatal(err)
	}
	defer catalog.Stop()
	manager, err := modelserver.New(modelserver.Config{
		ModelRoot: root, ContainerRoot: "/models", ReloadEndpoint: reload.URL,
		GRPCEndpoint: listener.Addr().String(), Catalog: catalog,
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.Start()
	defer manager.Stop()
	pipelines, err := pipeline.NewRegistry(pipeline.RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	// The platform ships no compiled-in graph; tests seed the sample
	// face-recognition definition through the public Replace channel.
	if err := pipelines.Replace([]pipeline.Definition{testsupport.FaceRecognitionDefinition()}); err != nil {
		t.Fatal(err)
	}
	backends := infer.NewRegistry("", nil)
	defer backends.Close()
	job := New("face-readiness-retry", Config{
		PipelineID: testsupport.FaceRecognitionID,
		SourceKind: "jpeg", SourceURL: camera.URL, Trigger: TriggerCommand,
		MaxInferMs: 5000, PublishMode: PublishAll,
	}, Deps{
		Catalog: catalog, Frames: mustNewManager(t), Backends: backends, Decoders: decode.NewRegistry(),
		Servables: modelserver.NewRegistry(manager), Pipelines: pipelines, Log: testLogger{t: t},
		JSONCap: 1 << 20, FrameTTLFactor: 2,
	})
	job.Start()
	commandRunning := true
	defer func() {
		if commandRunning {
			job.Stop()
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	if _, err := job.Invoke(ctx, ""); err != nil {
		t.Fatalf("resolve and invoke v1: %v", err)
	}
	initialDetectorReady, initialEmbeddingReady := fake.readyCallCounts()
	if initialDetectorReady != 1 || initialEmbeddingReady != 1 {
		t.Fatalf("initial readiness probes = %d/%d, want 1/1", initialDetectorReady, initialEmbeddingReady)
	}
	if active, latest := job.PipelineVersions(); active != "1.0.0" || latest != "1.0.0" {
		t.Fatalf("initial active/latest = %q/%q, want 1.0.0/1.0.0", active, latest)
	}

	// The new immutable graph uses the same packages, so only the server-ready
	// transition can make its resolution succeed. v1 must remain usable while
	// the detector reports Ready=false.
	fake.setReady(fake.detectorName, false)
	v1, _ := pipelines.Lookup(testsupport.FaceRecognitionID)
	v2 := testsupport.FaceRecognitionDefinition()
	v2.Version = "2.0.0"
	if err := pipelines.Replace([]pipeline.Definition{v1, v2}); err != nil {
		t.Fatal(err)
	}
	waitForPipelineCondition(t, 10*time.Second, func() bool {
		detectorReady, _ := fake.readyCallCounts()
		return detectorReady > initialDetectorReady && job.StatsSnapshot()["LastError"] == ReasonModelNotReady
	}, "initial v2 readiness failure")
	readings, err := job.Invoke(ctx, "")
	if err != nil {
		t.Fatalf("v1 did not remain active after failed v2 readiness probe: %v", err)
	}
	oldGraph, _ := decodePipelineReading(t, readings)
	if oldGraph.FrameID != 2 || len(oldGraph.Objects) != 1 {
		t.Fatalf("old graph result after failed upgrade = %+v", oldGraph)
	}
	if refs := job.PipelineModelRefs(); len(refs) != 2 || refs[0].Version != "1.0.0" || refs[1].Version != "1.0.0" {
		t.Fatalf("failed readiness upgrade replaced v1 package refs: %+v", refs)
	}
	if got := job.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady || got["SuccessTotal"] != int64(2) {
		t.Fatalf("old-graph success concealed the latest resolution failure: %+v", got)
	}
	if active, latest := job.PipelineVersions(); active != "1.0.0" || latest != "2.0.0" {
		t.Fatalf("failed-upgrade active/latest = %q/%q, want 1.0.0/2.0.0", active, latest)
	}

	// Keep v2 unavailable through another timed resolve retry. Another v1
	// command succeeds, but repeated NE-style health polls must remain FAILED.
	beforeRetryDetector, _ := fake.readyCallCounts()
	waitForPipelineCondition(t, 8*time.Second, func() bool {
		detectorReady, _ := fake.readyCallCounts()
		return detectorReady > beforeRetryDetector
	}, "second v2 readiness failure on the timed retry")
	if got := job.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady {
		t.Fatalf("timed retry concealed or changed the resolution failure: %+v", got)
	}
	if _, err := job.Invoke(ctx, ""); err != nil {
		t.Fatalf("v1 failed after repeated v2 retries: %v", err)
	}
	if got := job.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady || got["SuccessTotal"] != int64(3) {
		t.Fatalf("third old-graph success concealed the latest failure: %+v", got)
	}

	// From this point there is deliberately no catalog or pipeline registry
	// event. Only the five-second active-version retry can observe recovery.
	failedDetectorReady, failedEmbeddingReady := fake.readyCallCounts()
	fake.setReady(fake.detectorName, true)
	waitForPipelineCondition(t, 15*time.Second, func() bool {
		detectorReady, embeddingReady := fake.readyCallCounts()
		got := job.StatsSnapshot()
		return detectorReady > failedDetectorReady && embeddingReady > failedEmbeddingReady &&
			got["State"] == StateApplying && got["LastError"] == ""
	}, "v2 readiness recovery without a catalog or pipeline event")
	if got := job.StatsSnapshot(); got["State"] != StateApplying || got["LastError"] != "" {
		t.Fatalf("recovered v2 did not clear resolution failure: %+v", got)
	}
	if active, latest := job.PipelineVersions(); active != "2.0.0" || latest != "2.0.0" {
		t.Fatalf("recovered active/latest = %q/%q, want 2.0.0/2.0.0", active, latest)
	}
	readings, err = job.Invoke(ctx, "")
	if err != nil {
		t.Fatalf("invoke recovered graph: %v", err)
	}
	recovered, _ := decodePipelineReading(t, readings)
	if recovered.FrameID != 4 || len(recovered.Objects) != 1 {
		t.Fatalf("recovered graph result = %+v", recovered)
	}

	// A same-version re-resolution can also fail transiently. Active==latest
	// must not suppress timed recovery while the resolution latch is pending.
	fake.setReady(fake.detectorName, false)
	if err := pipelines.Replace([]pipeline.Definition{v1, v2}); err != nil {
		t.Fatal(err)
	}
	waitForPipelineCondition(t, 10*time.Second, func() bool {
		active, latest := job.PipelineVersions()
		return active == "2.0.0" && latest == "2.0.0" && job.StatsSnapshot()["LastError"] == ReasonModelNotReady
	}, "same-version v2 readiness failure")
	if _, err := job.Invoke(ctx, ""); err != nil {
		t.Fatalf("active v2 did not survive same-version re-resolution failure: %v", err)
	}
	if got := job.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady {
		t.Fatalf("same-version old-graph success concealed failure: %+v", got)
	}
	beforeSameVersionRecovery, _ := fake.readyCallCounts()
	fake.setReady(fake.detectorName, true)
	waitForPipelineCondition(t, 15*time.Second, func() bool {
		detectorReady, _ := fake.readyCallCounts()
		got := job.StatsSnapshot()
		return detectorReady > beforeSameVersionRecovery && got["State"] == StateApplying && got["LastError"] == ""
	}, "same-version timed readiness recovery")

	// A continuous Job shares the resolved v2 packages but has its own executor
	// and health. Stop the command Job so subsequent readiness probes can be
	// attributed to the continuous retry loop alone.
	job.Stop()
	commandRunning = false
	continuous := New("face-readiness-continuous", Config{
		PipelineID: testsupport.FaceRecognitionID,
		SourceKind: "jpeg", SourceURL: camera.URL, Trigger: TriggerContinuous, FPS: 20,
		MaxInferMs: 5000, PublishMode: PublishNone,
	}, Deps{
		Catalog: catalog, Frames: mustNewManager(t), Backends: backends, Decoders: decode.NewRegistry(),
		Servables: modelserver.NewRegistry(manager), Pipelines: pipelines, Log: testLogger{t: t},
		JSONCap: 1 << 20, FrameTTLFactor: 2,
	})
	continuous.Start()
	defer continuous.Stop()
	waitForPipelineCondition(t, 10*time.Second, func() bool {
		active, latest := continuous.PipelineVersions()
		return active == "2.0.0" && latest == "2.0.0" && continuous.StatsSnapshot()["SuccessTotal"].(int64) > 0
	}, "continuous v2 baseline")

	fake.setReady(fake.detectorName, false)
	v3 := testsupport.FaceRecognitionDefinition()
	v3.Version = "3.0.0"
	if err := pipelines.Replace([]pipeline.Definition{v1, v2, v3}); err != nil {
		t.Fatal(err)
	}
	waitForPipelineCondition(t, 10*time.Second, func() bool {
		active, latest := continuous.PipelineVersions()
		return active == "2.0.0" && latest == "3.0.0" && continuous.StatsSnapshot()["LastError"] == ReasonModelNotReady
	}, "continuous failed v3 resolution")
	beforeSuccess := continuous.StatsSnapshot()["SuccessTotal"].(int64)
	waitForPipelineCondition(t, 5*time.Second, func() bool {
		return continuous.StatsSnapshot()["SuccessTotal"].(int64) > beforeSuccess
	}, "continuous old-graph success during failed upgrade")
	for poll := 0; poll < 3; poll++ {
		if got := continuous.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady {
			t.Fatalf("continuous health poll %d concealed latest failure: %+v", poll, got)
		}
	}
	beforeContinuousRetry, _ := fake.readyCallCounts()
	waitForPipelineCondition(t, 8*time.Second, func() bool {
		detectorReady, _ := fake.readyCallCounts()
		return detectorReady > beforeContinuousRetry
	}, "continuous repeated v3 resolve retry")
	if got := continuous.StatsSnapshot(); got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady {
		t.Fatalf("continuous successes cleared failure across retries: %+v", got)
	}

	fake.setReady(fake.detectorName, true)
	waitForPipelineCondition(t, 15*time.Second, func() bool {
		active, latest := continuous.PipelineVersions()
		got := continuous.StatsSnapshot()
		return active == "3.0.0" && latest == "3.0.0" && got["State"] == StateApplying && got["LastError"] == ""
	}, "continuous v3 recovery")
}

func decodePipelineReading(t *testing.T, readings []decode.Reading) (pipeline.Result, string) {
	t.Helper()
	for _, reading := range readings {
		if reading.Name != "PipelineJSON" {
			continue
		}
		raw, ok := reading.Value.(string)
		if !ok {
			t.Fatalf("PipelineJSON value has type %T", reading.Value)
		}
		var result pipeline.Result
		if err := json.Unmarshal([]byte(raw), &result); err != nil {
			t.Fatalf("invalid PipelineJSON %q: %v", raw, err)
		}
		return result, raw
	}
	t.Fatalf("PipelineJSON missing from readings: %+v", readings)
	return pipeline.Result{}, ""
}

func assertEmitterPipelinePayload(t *testing.T, events []recordedEmission, index int, want string) {
	t.Helper()
	if len(events) <= index {
		t.Fatalf("emitter has %d events, want index %d", len(events), index)
	}
	for _, reading := range events[index].readings {
		if reading.Name == "PipelineJSON" && reading.Value == want {
			return
		}
	}
	t.Fatalf("event %d did not publish PipelineJSON to the local bus: %+v", index, events[index])
}

func waitForPipelineCondition(t *testing.T, timeout time.Duration, condition func() bool, description string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s; stats=%+v", description, (&Stats{State: StateApplying}).Snapshot())
}

func equalInt64s(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func facePipelineYuNetResponse(score float32) *pb.ModelInferResponse {
	outputs := make([]*pb.ModelInferResponse_InferOutputTensor, 0, 12)
	for _, stride := range []int64{8, 16, 32} {
		anchors := (64 / stride) * (64 / stride)
		for _, role := range []struct {
			name     string
			channels int64
		}{{"cls", 1}, {"obj", 1}, {"bbox", 4}, {"kps", 10}} {
			data := make([]float32, anchors*role.channels)
			outputs = append(outputs, &pb.ModelInferResponse_InferOutputTensor{
				Name: fmt.Sprintf("%s_%d", role.name, stride), Datatype: "FP32", Shape: []int64{1, anchors, role.channels},
				Contents: &pb.InferTensorContents{Fp32Contents: data},
			})
		}
	}
	if score > 0 {
		byName := func(name string) *pb.ModelInferResponse_InferOutputTensor {
			for _, output := range outputs {
				if output.GetName() == name {
					return output
				}
			}
			return nil
		}
		byName("cls_8").Contents.Fp32Contents[0] = score
		byName("obj_8").Contents.Fp32Contents[0] = score
		bbox := byName("bbox_8").Contents.Fp32Contents
		bbox[0], bbox[1] = 3, 3
		bbox[2], bbox[3] = float32(math.Log(2)), float32(math.Log(3))
		points := [5][2]float32{{20, 20}, {28, 20}, {24, 24}, {20, 28}, {28, 28}}
		for i, point := range points {
			byName("kps_8").Contents.Fp32Contents[2*i] = point[0] / 8
			byName("kps_8").Contents.Fp32Contents[2*i+1] = point[1] / 8
		}
	}
	return &pb.ModelInferResponse{Outputs: outputs}
}

func writeFacePipelinePackage(t *testing.T, packagePath, id string) {
	t.Helper()
	payload := []byte("synthetic ONNX payload for " + id)
	payloadHash := sha256.Sum256(payload)
	manifest := facePipelineManifest(id, payload, hex.EncodeToString(payloadHash[:]))
	file, err := os.Create(packagePath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gzipWriter := gzip.NewWriter(file)
	tarWriter := tar.NewWriter(gzipWriter)
	files := map[string][]byte{
		"nereid-model.json":                  manifest,
		"repository/" + id + "/1/model.onnx": payload,
	}
	for _, name := range []string{"nereid-model.json", "repository/" + id + "/1/model.onnx"} {
		if err := tarWriter.WriteHeader(&tar.Header{Name: name, Size: int64(len(files[name])), Mode: 0o644, Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tarWriter.Write(files[name]); err != nil {
			t.Fatal(err)
		}
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}
}

func facePipelineManifest(id string, payload []byte, payloadHash string) []byte {
	manifest := map[string]any{
		"schema": "nereid-model/1",
		"model":  map[string]any{"id": id, "version": "1.0.0", "task": "detection"},
		"artifact": map[string]any{"format": "onnx", "precision": "FP32", "files": []any{map[string]any{
			"path": "repository/" + id + "/1/model.onnx", "bytes": len(payload), "sha256": payloadHash,
		}}},
		"runtime": map[string]any{
			"backend": "kserve", "model_name": id + "-manifest-name", "repository_path": "repository/" + id,
			"server": map[string]any{"family": "ovms"},
		},
	}
	if id == "yunet" {
		outputs := make([]any, 0, 12)
		strides := make([]any, 0, 3)
		for _, stride := range []int64{8, 16, 32} {
			anchors := (64 / stride) * (64 / stride)
			strideSpec := map[string]any{"stride": stride}
			for _, role := range []struct {
				name     string
				channels int64
			}{{"cls", 1}, {"obj", 1}, {"bbox", 4}, {"kps", 10}} {
				name := fmt.Sprintf("%s_%d", role.name, stride)
				strideSpec[role.name] = name
				outputs = append(outputs, map[string]any{"name": name, "shape": []int64{1, anchors, role.channels}, "dtype": "FP32"})
			}
			strides = append(strides, strideSpec)
		}
		manifest["io"] = map[string]any{
			"inputs": []any{map[string]any{
				"name": "yunet-input", "shape": []int64{1, 3, 64, 64}, "layout": "NCHW", "dtype": "FP32", "color_order": "RGB",
				"preprocess": map[string]any{"resize": "stretch", "scale": "1/255"},
			}},
			"outputs": outputs,
		}
		manifest["decoder"] = map[string]any{
			"kind": "detection.yunet", "threshold": 0.5, "activation": "probability",
			"nms": map[string]any{"iou": 0.3, "top_k": 300}, "strides": strides,
		}
	} else {
		manifest["model"] = map[string]any{"id": id, "version": "1.0.0", "task": "embeddings"}
		manifest["io"] = map[string]any{
			"inputs": []any{map[string]any{
				"name": "sface-input", "shape": []int64{1, 3, 112, 112}, "layout": "NCHW", "dtype": "FP32", "color_order": "RGB",
				"preprocess": map[string]any{
					"resize": "stretch", "scale": "1/255",
					"alignment": map[string]any{"template": [][]float32{{38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041}}},
				},
			}},
			"outputs": []any{map[string]any{"name": "embedding", "shape": []int64{1, 128}, "dtype": "FP32"}},
		}
		manifest["decoder"] = map[string]any{"kind": "embeddings.vector", "output": "embedding", "l2_normalize": false, "max_dims": 128}
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		panic(err)
	}
	return raw
}
