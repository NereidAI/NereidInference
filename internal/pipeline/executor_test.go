// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
)

type pipelineTestCandidate struct {
	anchor       int
	score        float32
	cx, cy, w, h float32
}

type pipelineTestBackend struct {
	mu             sync.Mutex
	detectorCalls  int
	embeddingCalls int
	embeddingDims  int
	nonFinite      bool
	shapes         [][]int64
	modelNames     []string
	response       *infer.Response
}

func (b *pipelineTestBackend) Kind() string                                { return "kserve" }
func (b *pipelineTestBackend) Ready(context.Context, string, string) error { return nil }
func (b *pipelineTestBackend) Infer(_ context.Context, req infer.Request) (*infer.Response, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	switch req.ModelName {
	case "yunet-wire":
		b.detectorCalls++
		return b.response, nil
	case "sface-wire":
		b.embeddingCalls++
		b.shapes = append(b.shapes, append([]int64(nil), req.Inputs[0].Shape...))
		b.modelNames = append(b.modelNames, req.ModelName)
		if len(req.Inputs) != 1 || req.Inputs[0].Name != "sface-input" {
			return nil, fmt.Errorf("unexpected SFace inputs: %+v", req.Inputs)
		}
		dims := b.embeddingDims
		if dims == 0 {
			dims = 128
		}
		vector := make([]float32, dims)
		for i := range vector {
			vector[i] = float32(i+1) / 128
		}
		if b.nonFinite {
			vector[3] = float32(math.NaN())
		}
		return &infer.Response{Outputs: map[string]*infer.TensorOutput{
			"embedding": {Name: "embedding", Shape: []int64{1, int64(dims)}, Float32: vector},
		}}, nil
	default:
		return nil, fmt.Errorf("unexpected model %q", req.ModelName)
	}
}

func TestExecutorZeroOneAndNFacesWithSerialBatchOne(t *testing.T) {
	for _, tc := range []struct {
		name           string
		candidates     []pipelineTestCandidate
		maxObjects     int
		wantObjects    int
		wantSFaceCalls int
		wantDropped    int
	}{
		{name: "zero", wantObjects: 0, wantSFaceCalls: 0, maxObjects: 8},
		{name: "one", candidates: []pipelineTestCandidate{{anchor: 0, score: 0.95, cx: 8, cy: 8, w: 12, h: 12}}, maxObjects: 8, wantObjects: 1, wantSFaceCalls: 1},
		{name: "N capped before SFace", candidates: []pipelineTestCandidate{
			{anchor: 0, score: 0.7, cx: 8, cy: 8, w: 10, h: 10},
			{anchor: 2, score: 0.95, cx: 24, cy: 8, w: 10, h: 10},
			{anchor: 4, score: 0.9, cx: 40, cy: 8, w: 10, h: 10},
		}, maxObjects: 2, wantObjects: 2, wantSFaceCalls: 2, wantDropped: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &pipelineTestBackend{response: pipelineYuNetResponse(64, tc.candidates)}
			executor := newTestExecutor(backend, tc.maxObjects)
			result, err := executor.Execute(context.Background(), pipelineTestFrame(64), 0)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Objects) != tc.wantObjects || backend.embeddingCalls != tc.wantSFaceCalls {
				t.Fatalf("objects=%d SFace calls=%d, want %d/%d", len(result.Objects), backend.embeddingCalls, tc.wantObjects, tc.wantSFaceCalls)
			}
			if result.DroppedObjects != tc.wantDropped {
				t.Fatalf("dropped = %d, want %d", result.DroppedObjects, tc.wantDropped)
			}
			if result.Objects == nil {
				t.Fatal("objects must never be null")
			}
			for _, shape := range backend.shapes {
				if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != 2 || shape[3] != 2 {
					t.Fatalf("SFace call was not batch=1 [1,3,2,2]: %v", shape)
				}
			}
			for i, object := range result.Objects {
				if object.ID != int64(i+1) || object.ParentID != 0 || object.Type != "face" {
					t.Fatalf("lineage/type lost: %+v", object)
				}
				if len(object.Keypoints) != 5 || len(object.Embedding) != 128 {
					t.Fatalf("face products missing: %+v", object)
				}
			}
			readings, err := result.Readings(1 << 20)
			if err != nil {
				t.Fatal(err)
			}
			payload := readings[0].Value.(string)
			if tc.wantObjects == 0 && !strings.Contains(payload, `"objects":[]`) {
				t.Fatalf("healthy empty event = %s", payload)
			}
			if strings.Contains(payload, "tensor") || strings.Contains(payload, "jpeg") || strings.Contains(payload, "Float32") {
				t.Fatalf("raw image/tensor leaked into PipelineJSON: %s", payload)
			}
		})
	}
}

func TestExecutorRejectsWrongEmbeddingDimensionsAndNonFiniteValues(t *testing.T) {
	candidates := []pipelineTestCandidate{{anchor: 0, score: 0.95, cx: 8, cy: 8, w: 12, h: 12}}
	for _, tc := range []struct {
		name      string
		dims      int
		nonFinite bool
	}{{name: "wrong dimensions", dims: 64}, {name: "non-finite", dims: 128, nonFinite: true}} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &pipelineTestBackend{response: pipelineYuNetResponse(64, candidates), embeddingDims: tc.dims, nonFinite: tc.nonFinite}
			executor := newTestExecutor(backend, 8)
			if _, err := executor.Execute(context.Background(), pipelineTestFrame(64), 0); err == nil {
				t.Fatal("expected invalid SFace output to fail the frame")
			}
		})
	}
}

func TestPipelineResultEventCapDropsTailDeterministically(t *testing.T) {
	objects := make([]Object, 3)
	for i := range objects {
		embedding := make([]float32, 128)
		for j := range embedding {
			embedding[j] = 0.123456789
		}
		objects[i] = Object{ID: int64(i + 1), Type: "face", Score: float32(3 - i), Keypoints: []decode.Keypoint{}, Embedding: embedding}
	}
	result := &Result{Schema: "nereid.pipeline/1", PipelineID: "face", FrameID: 9, Objects: objects, maxEventBytes: 3000}
	readings, err := result.Readings(1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	payload := readings[0].Value.(string)
	if len(payload) > 3000 {
		t.Fatalf("payload = %d bytes, cap 3000", len(payload))
	}
	var document Result
	if err := json.Unmarshal([]byte(payload), &document); err != nil {
		t.Fatal(err)
	}
	if len(document.Objects) >= len(objects) || document.DroppedObjects == 0 {
		t.Fatalf("cap did not deterministically drop/count tail objects: %s", payload)
	}
	for i, object := range document.Objects {
		if object.ID != int64(i+1) {
			t.Fatalf("cap did not retain the highest-score prefix: %+v", document.Objects)
		}
	}
	tooSmall := &Result{Schema: "nereid.pipeline/1", PipelineID: "face", Objects: []Object{}, maxEventBytes: 10}
	if _, err := tooSmall.Readings(10); err == nil {
		t.Fatal("an impossible empty-event cap must fail explicitly")
	}
}

func TestExecutorResolveAcquiresTwoLeasesAndReleasesThem(t *testing.T) {
	detectorEntry := &modelcatalog.Entry{ID: "yunet", Version: "1.0.0", Manifest: pipelineDetectorManifest(64)}
	embeddingEntry := &modelcatalog.Entry{ID: "sface", Version: "1.0.0", Manifest: pipelineEmbeddingManifest()}
	catalog := pipelineTestCatalog{"yunet": detectorEntry, "sface": embeddingEntry}
	backend := &pipelineTestBackend{}
	manager := &pipelineTestServableManager{}
	servables := modelserver.NewRegistry(manager)
	executor := NewExecutor(ExecutorDeps{
		Catalog: catalog, Backends: pipelineTestBackendRegistry{backend: backend}, Decoders: decode.NewRegistry(), Servables: servables,
	})
	definition := faceRecognitionDefinition()
	definition.Nodes[1].Model.TargetDevice = "GPU.1"
	if err := executor.Resolve(context.Background(), definition, "CPU", ""); err != nil {
		t.Fatal(err)
	}
	if len(manager.leases) != 2 {
		t.Fatalf("Acquire calls = %d, want 2", len(manager.leases))
	}
	if manager.leases[0].target != "GPU.1" || manager.leases[1].target != "CPU" {
		t.Fatalf("graph override/default targets = %q/%q, want GPU.1/CPU",
			manager.leases[0].target, manager.leases[1].target)
	}
	executor.Reset()
	for _, lease := range manager.leases {
		if lease.releases != 1 {
			t.Fatalf("lease releases = %d, want 1", lease.releases)
		}
	}
}

func TestExecutorResolveRequiresManagedLeases(t *testing.T) {
	catalog := pipelineTestCatalog{
		"yunet": {ID: "yunet", Version: "1.0.0", Manifest: pipelineDetectorManifest(64)},
		"sface": {ID: "sface", Version: "1.0.0", Manifest: pipelineEmbeddingManifest()},
	}
	for _, servables := range []*modelserver.Registry{nil, modelserver.NewRegistry()} {
		executor := NewExecutor(ExecutorDeps{
			Catalog: catalog, Backends: pipelineTestBackendRegistry{backend: &pipelineTestBackend{}},
			Decoders: decode.NewRegistry(), Servables: servables,
		})
		if err := executor.Resolve(context.Background(), faceRecognitionDefinition(), "CPU", ""); err == nil {
			t.Fatal("managed model resolved without its required servable lease")
		}
	}
}

func TestExecutorResolveRejectsExternalWithoutManagedFallback(t *testing.T) {
	external := pipelineDetectorManifest(64)
	external.Runtime.Server.Family = modelserver.FamilyExternal
	catalog := pipelineTestCatalog{
		"yunet": {ID: "yunet", Version: "1.0.0", Manifest: external},
		"sface": {ID: "sface", Version: "1.0.0", Manifest: pipelineEmbeddingManifest()},
	}
	backend := &recordingPipelineBackendRegistry{backend: &pipelineTestBackend{}}
	manager := &pipelineTestServableManager{}
	executor := NewExecutor(ExecutorDeps{
		Catalog: catalog, Backends: backend, Decoders: decode.NewRegistry(),
		Servables: modelserver.NewRegistry(manager),
	})
	err := executor.Resolve(context.Background(), faceRecognitionDefinition(), "CPU", "")
	var pipelineErr *Error
	if !errors.As(err, &pipelineErr) || pipelineErr.Reason != ReasonConfigInvalid {
		t.Fatalf("external Resolve error = %v, want config-invalid", err)
	}
	if len(manager.leases) != 0 || len(backend.endpoints) != 0 || executor.ResolvedVersion() != "" {
		t.Fatalf("external family fell back to local execution: leases=%v endpoints=%v active=%q",
			manager.leases, backend.endpoints, executor.ResolvedVersion())
	}
}

func TestExecutorResolveUsesEachManagedLeaseEndpoint(t *testing.T) {
	detectorManifest := pipelineDetectorManifest(64)
	embeddingManifest := pipelineEmbeddingManifest()
	embeddingManifest.Runtime.Server.Family = modelserver.FamilyTriton
	catalog := pipelineTestCatalog{
		"yunet": {ID: "yunet", Version: "1.0.0", Manifest: detectorManifest},
		"sface": {ID: "sface", Version: "1.0.0", Manifest: embeddingManifest},
	}
	backend := &recordingPipelineBackendRegistry{backend: &pipelineTestBackend{}}
	servables := modelserver.NewRegistry(
		&pipelineTestServableManager{endpoint: "ovms:9000"},
		&pipelineTestServableManager{family: modelserver.FamilyTriton, endpoint: "triton:8001"},
	)
	executor := NewExecutor(ExecutorDeps{
		Catalog: catalog, Backends: backend, Decoders: decode.NewRegistry(), Servables: servables,
	})
	if err := executor.Resolve(context.Background(), faceRecognitionDefinition(), "CPU", ""); err != nil {
		t.Fatal(err)
	}
	defer executor.Reset()
	if got := backend.endpoints; len(got) != 2 || got[0] != "ovms:9000" || got[1] != "triton:8001" {
		t.Fatalf("backend endpoints = %v, want the two managed lease endpoints", got)
	}
}

func TestExecutorResolveRejectsEmptyLeaseEndpointEvenWithOverride(t *testing.T) {
	catalog := pipelineTestCatalog{
		"yunet": {ID: "yunet", Version: "1.0.0", Manifest: pipelineDetectorManifest(64)},
		"sface": {ID: "sface", Version: "1.0.0", Manifest: pipelineEmbeddingManifest()},
	}
	backend := &recordingPipelineBackendRegistry{backend: &pipelineTestBackend{}}
	manager := &pipelineTestServableManager{emptyEndpoint: true}
	executor := NewExecutor(ExecutorDeps{
		Catalog: catalog, Backends: backend, Decoders: decode.NewRegistry(),
		Servables: modelserver.NewRegistry(manager),
	})
	if err := executor.Resolve(context.Background(), faceRecognitionDefinition(), "CPU", "explicit:9000"); err == nil {
		t.Fatal("empty managed lease endpoint must fail even with an explicit job override")
	}
	if len(backend.endpoints) != 0 {
		t.Fatalf("backend was called despite invalid lease endpoint: %v", backend.endpoints)
	}
	if len(manager.leases) != 1 || manager.leases[0].releases != 1 {
		t.Fatalf("failed endpoint validation did not release acquired lease: %+v", manager.leases)
	}
}

func TestExecutorModelRefsDescribeActiveGraph(t *testing.T) {
	executor := newTestExecutor(&pipelineTestBackend{}, 8)
	refs := executor.ModelRefs()
	if len(refs) != 2 || refs[0].ID != "yunet" || refs[0].Version != "1.0.0" || refs[1].ID != "sface" {
		t.Fatalf("active model refs = %+v", refs)
	}
	if !executor.ReferencesModelVersion("yunet", "1.0.0") || executor.ReferencesModelVersion("yunet", "2.0.0") {
		t.Fatal("exact active package matching is incorrect")
	}
}

func newTestExecutor(backend *pipelineTestBackend, maxObjects int) *Executor {
	definition := normalizeDefinition(faceRecognitionDefinition())
	definition.Limits.MaxObjects = maxObjects
	return &Executor{current: &resolvedGraph{
		definition: definition,
		detector: &resolvedNode{
			node: definition.Nodes[1], entry: &modelcatalog.Entry{ID: "yunet", Version: "1.0.0", Manifest: pipelineDetectorManifest(64)},
			decoder: mustPipelineDecoder(pipelineDetectorManifest(64)), backend: backend, modelName: "yunet-wire",
		},
		embedding: &resolvedNode{
			node: definition.Nodes[3], entry: &modelcatalog.Entry{ID: "sface", Version: "1.0.0", Manifest: pipelineEmbeddingManifest()},
			decoder: mustPipelineDecoder(pipelineEmbeddingManifest()), backend: backend, modelName: "sface-wire",
		},
	}}
}

func mustPipelineDecoder(manifest *model.Manifest) decode.Decoder {
	decoder, err := decode.NewRegistry().Build(manifest)
	if err != nil {
		panic(err)
	}
	return decoder
}

func pipelineDetectorManifest(input int) *model.Manifest {
	strides := make([]map[string]any, 0, 3)
	for _, stride := range []int{8, 16, 32} {
		strides = append(strides, map[string]any{
			"stride": stride, "cls": fmt.Sprintf("cls_%d", stride), "obj": fmt.Sprintf("obj_%d", stride),
			"bbox": fmt.Sprintf("bbox_%d", stride), "kps": fmt.Sprintf("kps_%d", stride),
		})
	}
	decoder, _ := json.Marshal(map[string]any{"kind": "detection.yunet", "threshold": 0.5, "strides": strides})
	return &model.Manifest{
		Model: model.ModelSection{ID: "yunet", Version: "1.0.0", Task: "detection"},
		Runtime: model.RuntimeSection{Backend: "kserve", ModelName: "yunet-wire", Server: struct {
			Family     string `json:"family"`
			Image      string `json:"image"`
			MinVersion string `json:"min_version"`
		}{Family: "ovms"}},
		IO: model.IOSection{Inputs: []model.IOSpec{{
			Name: "yunet-input", Shape: []int64{1, 3, int64(input), int64(input)}, Layout: "NCHW", DType: "FP32",
			Preprocess: &model.PreprocessSpec{Resize: "stretch", Scale: "1/255"},
		}}},
		Decoder: decoder,
	}
}

func pipelineEmbeddingManifest() *model.Manifest {
	return &model.Manifest{
		Model: model.ModelSection{ID: "sface", Version: "1.0.0", Task: "embeddings"},
		Runtime: model.RuntimeSection{Backend: "kserve", ModelName: "sface-wire", Server: struct {
			Family     string `json:"family"`
			Image      string `json:"image"`
			MinVersion string `json:"min_version"`
		}{Family: "ovms"}},
		IO: model.IOSection{Inputs: []model.IOSpec{{
			Name: "sface-input", Shape: []int64{1, 3, 2, 2}, Layout: "NCHW", DType: "FP32", ColorOrder: "RGB",
			Preprocess: &model.PreprocessSpec{
				Resize: "stretch", Scale: "1/255",
				Alignment: &model.AlignmentSpec{Template: [][2]float32{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}},
			},
		}}},
		Decoder: json.RawMessage(`{"kind":"embeddings.vector","output":"embedding","l2_normalize":false}`),
	}
}

func pipelineYuNetResponse(input int, candidates []pipelineTestCandidate) *infer.Response {
	response := &infer.Response{Outputs: map[string]*infer.TensorOutput{}}
	for _, stride := range []int{8, 16, 32} {
		anchors := (input / stride) * (input / stride)
		for _, role := range []struct {
			name     string
			channels int
			score    bool
		}{{"cls", 1, true}, {"obj", 1, true}, {"bbox", 4, false}, {"kps", 10, false}} {
			name := fmt.Sprintf("%s_%d", role.name, stride)
			data := make([]float32, anchors*role.channels)
			if role.score {
				for i := range data {
					data[i] = -20
				}
			}
			response.Outputs[name] = &infer.TensorOutput{Name: name, Shape: []int64{1, int64(anchors), int64(role.channels)}, Float32: data}
		}
	}
	for _, candidate := range candidates {
		const stride = 8
		cols := input / stride
		x := candidate.anchor % cols
		response.Outputs["cls_8"].Float32[candidate.anchor] = candidate.score
		response.Outputs["obj_8"].Float32[candidate.anchor] = candidate.score
		bbox := response.Outputs["bbox_8"].Float32[candidate.anchor*4:]
		bbox[0] = candidate.cx/stride - float32(x)
		bbox[1] = candidate.cy / stride
		bbox[2] = float32(math.Log(float64(candidate.w / stride)))
		bbox[3] = float32(math.Log(float64(candidate.h / stride)))
		points := [5][2]float32{{candidate.cx - 2, candidate.cy - 2}, {candidate.cx + 2, candidate.cy - 2}, {candidate.cx, candidate.cy}, {candidate.cx - 2, candidate.cy + 2}, {candidate.cx + 2, candidate.cy + 2}}
		kps := response.Outputs["kps_8"].Float32[candidate.anchor*10:]
		for i, point := range points {
			kps[2*i] = point[0]/stride - float32(x)
			kps[2*i+1] = point[1] / stride
		}
	}
	return response
}

func pipelineTestFrame(size int) *frame.Frame {
	return &frame.Frame{Encoding: frame.RGB24, Width: size, Height: size, Bytes: make([]byte, size*size*3)}
}

type pipelineTestCatalog map[string]*modelcatalog.Entry

func (c pipelineTestCatalog) Lookup(id, version string) (*modelcatalog.Entry, bool) {
	entry, ok := c[id]
	if !ok || (version != "" && entry.Version != version) {
		return nil, false
	}
	return entry, true
}

type pipelineTestBackendRegistry struct{ backend infer.Backend }

func (r pipelineTestBackendRegistry) Get(string, string) (infer.Backend, error) {
	return r.backend, nil
}

type recordingPipelineBackendRegistry struct {
	backend   infer.Backend
	endpoints []string
}

func (r *recordingPipelineBackendRegistry) Get(_ string, endpoint string) (infer.Backend, error) {
	r.endpoints = append(r.endpoints, endpoint)
	return r.backend, nil
}

type pipelineTestLease struct {
	name, endpoint, target string
	releases               int
}

func (l *pipelineTestLease) ModelName() string         { return l.name }
func (l *pipelineTestLease) Endpoint() string          { return l.endpoint }
func (l *pipelineTestLease) Ref() modelserver.ModelRef { return modelserver.ModelRef{} }
func (l *pipelineTestLease) Current() bool             { return true }
func (l *pipelineTestLease) Release()                  { l.releases++ }

type pipelineTestServableManager struct {
	family        string
	endpoint      string
	emptyEndpoint bool
	leases        []*pipelineTestLease
}

func (m *pipelineTestServableManager) Family() string {
	if m.family != "" {
		return m.family
	}
	return modelserver.FamilyOVMS
}
func (m *pipelineTestServableManager) Acquire(_ context.Context, entry *modelcatalog.Entry, target string) (modelserver.Lease, error) {
	endpoint := m.endpoint
	if endpoint == "" && !m.emptyEndpoint {
		endpoint = "lease:9000"
	}
	lease := &pipelineTestLease{name: entry.Manifest.Runtime.ModelName, endpoint: endpoint, target: target}
	m.leases = append(m.leases, lease)
	return lease, nil
}
func (m *pipelineTestServableManager) RemoveModel(string, string) {}
func (m *pipelineTestServableManager) Start()                     {}
func (m *pipelineTestServableManager) Stop()                      {}
