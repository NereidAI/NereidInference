// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"sort"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

// Failure reasons match the job package's transport-safe vocabulary without
// introducing an import cycle.
const (
	ReasonConfigInvalid   = "config-invalid"
	ReasonModelNotReady   = "model-not-ready"
	ReasonRuntimeError    = "runtime-error"
	ReasonPreprocessFail  = "preprocess-failed"
	ReasonPostprocessFail = "postprocess-failed"
)

// Error carries a stable failure reason and the failing graph node.
type Error struct {
	Reason string
	Node   string
	Err    error
}

func (e *Error) Error() string {
	if e.Node == "" {
		return e.Err.Error()
	}
	return fmt.Sprintf("pipeline node %s: %v", e.Node, e.Err)
}

func (e *Error) Unwrap() error { return e.Err }

// Catalog is the read-only package resolution surface used by Executor.
type Catalog interface {
	Lookup(id, version string) (*modelcatalog.Entry, bool)
}

// BackendRegistry resolves the descriptor-selected wire backend.
type BackendRegistry interface {
	Get(kind, endpoint string) (infer.Backend, error)
}

// DecoderRegistry builds a codec solely from the model manifest.
type DecoderRegistry interface {
	Build(m *model.Manifest) (decode.Decoder, error)
}

// ExecutorDeps are shared NI services; no model server containers or goroutines
// are created here. Leases manage models inside deployment-owned servers.
type ExecutorDeps struct {
	Catalog   Catalog
	Backends  BackendRegistry
	Decoders  DecoderRegistry
	Servables *modelserver.Registry
}

// Executor holds the two resolved model stages for one Job. A Job executes
// frames serially, so no cross-frame item state or locking is required.
type Executor struct {
	deps    ExecutorDeps
	current *resolvedGraph
	frameID uint64
}

type resolvedGraph struct {
	definition Definition
	detector   *resolvedNode
	embedding  *resolvedNode
}

type resolvedNode struct {
	node      Node
	entry     *modelcatalog.Entry
	decoder   decode.Decoder
	backend   infer.Backend
	lease     modelserver.Lease
	modelName string
}

// Item is the typed frame-local object passed through the linear stages. Crop
// is internal only and is never serialized into PipelineJSON.
type Item struct {
	ID                 int64
	ParentID           int64
	Type               string
	Box                [4]float32
	Score              float32
	Keypoints          []decode.Keypoint
	alignmentKeypoints []decode.Keypoint
	Embedding          []float32
	Crop               *preprocess.Tensor
}

// NewExecutor creates an unresolved executor for one pipeline binding.
func NewExecutor(deps ExecutorDeps) *Executor { return &Executor{deps: deps} }

// Reset releases both model leases and makes the executor unavailable. It is
// used when a referenced package or pipeline version is removed.
func (e *Executor) Reset() {
	if e.current == nil {
		return
	}
	releaseNode(e.current.detector)
	releaseNode(e.current.embedding)
	e.current = nil
}

// ReferencesModel reports whether the active definition references id.
func (e *Executor) ReferencesModel(id string) bool {
	if e.current == nil {
		return false
	}
	return e.current.definition.Nodes[1].Model.ModelID == id || e.current.definition.Nodes[3].Model.ModelID == id
}

// ReferencesModelVersion reports whether an actually resolved stage uses the
// exact removed package. It deliberately examines the active graph, not the
// newest registered definition after a failed upgrade.
func (e *Executor) ReferencesModelVersion(id, version string) bool {
	if e.current == nil {
		return false
	}
	for _, node := range []*resolvedNode{e.current.detector, e.current.embedding} {
		if node != nil && node.entry != nil && node.entry.ID == id && node.entry.Version == version {
			return true
		}
	}
	return false
}

// ModelRefs returns the two exact packages holding active leases. The driver
// uses this resolved state for inventory counts rather than guessing from the
// latest registered pipeline version.
func (e *Executor) ModelRefs() []modelserver.ModelRef {
	if e.current == nil {
		return nil
	}
	refs := make([]modelserver.ModelRef, 0, 2)
	for _, node := range []*resolvedNode{e.current.detector, e.current.embedding} {
		if node != nil && node.entry != nil {
			refs = append(refs, modelserver.ModelRef{ID: node.entry.ID, Version: node.entry.Version})
		}
	}
	return refs
}

// ResolvedVersion returns the active immutable definition version, or empty
// when no graph has resolved. Jobs use it to retry a failed newer version
// without reacquiring an already-current graph on every retry tick.
func (e *Executor) ResolvedVersion() string {
	if e.current == nil {
		return ""
	}
	return e.current.definition.Version
}

// Resolve transactionally resolves both packages, codecs, leases and backends.
// A failed replacement leaves the previous working graph intact; callers can
// explicitly Reset first for removal semantics.
func (e *Executor) Resolve(ctx context.Context, definition Definition, defaultTarget, endpointOverride string) error {
	if err := ValidateDefinition(normalizeDefinition(definition)); err != nil {
		return &Error{Reason: ReasonConfigInvalid, Err: err}
	}
	definition = normalizeDefinition(definition)
	if e.deps.Catalog == nil || e.deps.Backends == nil || e.deps.Decoders == nil {
		return &Error{Reason: ReasonConfigInvalid, Err: fmt.Errorf("pipeline executor dependencies are incomplete")}
	}
	next := &resolvedGraph{definition: cloneDefinition(definition)}
	cleanup := true
	defer func() {
		if cleanup {
			releaseNode(next.detector)
			releaseNode(next.embedding)
		}
	}()
	for i, nodeIndex := range []int{1, 3} {
		node := definition.Nodes[nodeIndex]
		entry, ok := e.deps.Catalog.Lookup(node.Model.ModelID, node.Model.ModelVersion)
		if !ok {
			return &Error{Reason: ReasonModelNotReady, Node: node.ID, Err: fmt.Errorf("model %s/%s is not installed", node.Model.ModelID, node.Model.ModelVersion)}
		}
		resolved, err := e.resolveNode(ctx, node, entry, defaultTarget, endpointOverride)
		if err != nil {
			return err
		}
		if i == 0 {
			if resolved.decoder.Kind() != "detection.yunet" {
				releaseNode(resolved)
				return &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("detector decoder must be detection.yunet, got %s", resolved.decoder.Kind())}
			}
			next.detector = resolved
		} else {
			if resolved.decoder.Kind() != "embeddings.vector" {
				releaseNode(resolved)
				return &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("embedding decoder must be embeddings.vector, got %s", resolved.decoder.Kind())}
			}
			if definition.Nodes[2].Mode == "five-point" {
				spec := resolved.entry.Manifest.IO.Inputs[0]
				if spec.Preprocess == nil || spec.Preprocess.Alignment == nil || len(spec.Preprocess.Alignment.Template) != 5 {
					releaseNode(resolved)
					return &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("five-point mode requires a five-point template in the SFace manifest")}
				}
			}
			next.embedding = resolved
		}
	}
	previous := e.current
	e.current = next
	cleanup = false
	if previous != nil {
		releaseNode(previous.detector)
		releaseNode(previous.embedding)
	}
	return nil
}

func (e *Executor) resolveNode(ctx context.Context, node Node, entry *modelcatalog.Entry, defaultTarget, endpointOverride string) (_ *resolvedNode, err error) {
	manifest := entry.Manifest
	if manifest == nil || manifest.Runtime.Backend != "kserve" || len(manifest.IO.Inputs) == 0 {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("pipeline infer nodes require a KServe vision manifest")}
	}
	input := manifest.IO.Inputs[0]
	if len(node.Inputs) == 1 && node.Inputs[0].Name != "" && node.Inputs[0].Name != input.Name {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("input name %q does not match manifest input %q", node.Inputs[0].Name, input.Name)}
	}
	decoder, err := e.deps.Decoders.Build(manifest)
	if err != nil {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: err}
	}
	resolved := &resolvedNode{node: node, entry: entry, decoder: decoder, modelName: manifest.Runtime.ModelName}
	defer func() {
		if err != nil {
			releaseNode(resolved)
		}
	}()
	target := node.Model.TargetDevice
	if target == "" {
		target = defaultTarget
	}
	family := manifest.Runtime.Server.Family
	if !modelserver.IsManagedFamily(family) {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("pipeline v1 requires a managed ovms or triton model family, got %q", family)}
	}
	if e.deps.Servables == nil {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("no servable registry is configured for managed family %q", family)}
	}
	lease, acquireErr := e.deps.Servables.Acquire(ctx, entry, target)
	if acquireErr != nil {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: acquireErr}
	}
	if lease == nil {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("managed family %q returned a nil lease", family)}
	}
	resolved.lease = lease
	resolved.modelName = lease.ModelName()

	endpoint := lease.Endpoint()
	if endpoint == "" {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: fmt.Errorf("managed family %q returned a lease without a backend endpoint", family)}
	}
	if endpointOverride != "" {
		// Preserve the existing explicit per-job override for operators that
		// deliberately reach a managed server through another address. Unlike
		// the old default fallback, this can never conceal an unconfigured lease.
		endpoint = endpointOverride
	}
	backend, err := e.deps.Backends.Get(manifest.Runtime.Backend, endpoint)
	if err != nil {
		return nil, &Error{Reason: ReasonConfigInvalid, Node: node.ID, Err: err}
	}
	resolved.backend = backend
	if err := backend.Ready(ctx, resolved.modelName, ""); err != nil {
		return nil, &Error{Reason: ReasonModelNotReady, Node: node.ID, Err: fmt.Errorf("model %s is not ready: %w", resolved.modelName, err)}
	}
	return resolved, nil
}

// Execute runs one source frame through detection, alignment and N serial
// batch=1 embedding calls. Zero detections skips alignment and SFace entirely.
func (e *Executor) Execute(ctx context.Context, source *frame.Frame, confidenceOverride float32) (*Result, error) {
	if e.current == nil {
		return nil, &Error{Reason: ReasonModelNotReady, Err: fmt.Errorf("pipeline is not resolved")}
	}
	e.frameID++
	items, dropped, err := e.executeItems(ctx, source, confidenceOverride)
	if err != nil {
		return nil, err
	}
	objects := make([]Object, 0, len(items))
	for _, item := range items {
		objects = append(objects, Object{
			ID: item.ID, ParentID: item.ParentID, Type: item.Type, Box: item.Box,
			Score: item.Score, Keypoints: append([]decode.Keypoint(nil), item.Keypoints...),
			Embedding: append([]float32(nil), item.Embedding...),
		})
	}
	return &Result{
		Schema: "nereid.pipeline/1", PipelineID: e.current.definition.PipelineID,
		FrameID: e.frameID, Objects: objects, DroppedObjects: dropped,
		maxEventBytes: e.current.definition.Limits.MaxEventBytes,
	}, nil
}

func (e *Executor) executeItems(ctx context.Context, source *frame.Frame, confidenceOverride float32) ([]Item, int, error) {
	graph := e.current
	detector := graph.detector
	tensor, err := preprocess.Build(source, detector.entry.Manifest.IO.Inputs[0])
	if err != nil {
		return nil, 0, &Error{Reason: ReasonPreprocessFail, Node: detector.node.ID, Err: err}
	}
	response, err := detector.backend.Infer(ctx, infer.Request{
		ModelName: detector.modelName,
		Inputs: []infer.TensorInput{{
			Name: detector.entry.Manifest.IO.Inputs[0].Name, Shape: tensor.Shape,
			Float32: tensor.Data, DataType: "FP32",
		}},
	})
	if err != nil {
		return nil, 0, &Error{Reason: ReasonRuntimeError, Node: detector.node.ID, Err: err}
	}
	decoded, err := detector.decoder.Decode(&decode.Input{
		Manifest: detector.entry.Manifest, Labels: detector.entry.Labels,
		Response: response, Meta: tensor.Meta, ConfOverride: confidenceOverride,
	})
	if err != nil {
		return nil, 0, &Error{Reason: ReasonPostprocessFail, Node: detector.node.ID, Err: err}
	}
	detections, ok := decoded.(*decode.DetectionResult)
	if !ok {
		return nil, 0, &Error{Reason: ReasonPostprocessFail, Node: detector.node.ID, Err: fmt.Errorf("decoder returned %T, want DetectionResult", decoded)}
	}
	ordered := append([]decode.Detection(nil), detections.Detections...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Score > ordered[j].Score })
	maxObjects := graph.definition.Limits.MaxObjects
	dropped := 0
	if len(ordered) > maxObjects {
		dropped = len(ordered) - maxObjects
		ordered = ordered[:maxObjects]
	}
	items := make([]Item, 0, len(ordered))
	for i, detection := range ordered {
		items = append(items, Item{
			ID: int64(i + 1), ParentID: 0, Type: "face", Box: detection.Box,
			Score: detection.Score, Keypoints: append([]decode.Keypoint(nil), detection.Keypoints...),
			alignmentKeypoints: append([]decode.Keypoint(nil), detection.KeypointsForAlignment()...),
		})
	}
	if len(items) == 0 {
		return items, dropped, nil
	}

	embedding := graph.embedding
	for i := range items {
		item := &items[i]
		var crop *preprocess.Tensor
		if graph.definition.Nodes[2].Mode == "roi" {
			crop, err = preprocess.BuildROI(source, embedding.entry.Manifest.IO.Inputs[0], item.Box)
		} else {
			points := make([][2]float32, len(item.alignmentKeypoints))
			for k, keypoint := range item.alignmentKeypoints {
				points[k] = [2]float32{keypoint.X, keypoint.Y}
			}
			crop, err = preprocess.BuildFivePoint(source, embedding.entry.Manifest.IO.Inputs[0], item.Box, points)
		}
		if err != nil {
			return nil, dropped, &Error{Reason: ReasonPreprocessFail, Node: graph.definition.Nodes[2].ID, Err: fmt.Errorf("object %d: %w", item.ID, err)}
		}
		// The crop is a child packet for lineage, while its vector is attached
		// back to the parent face object for emission.
		child := Item{ID: int64(len(items) + i + 1), ParentID: item.ID, Type: "face-crop", Box: item.Box, Crop: crop}
		response, err := embedding.backend.Infer(ctx, infer.Request{
			ModelName: embedding.modelName,
			Inputs: []infer.TensorInput{{
				Name: embedding.entry.Manifest.IO.Inputs[0].Name, Shape: child.Crop.Shape,
				Float32: child.Crop.Data, DataType: "FP32",
			}},
		})
		if err != nil {
			return nil, dropped, &Error{Reason: ReasonRuntimeError, Node: embedding.node.ID, Err: fmt.Errorf("object %d: %w", item.ID, err)}
		}
		decoded, err := embedding.decoder.Decode(&decode.Input{
			Manifest: embedding.entry.Manifest, Labels: embedding.entry.Labels,
			Response: response, Meta: crop.Meta,
		})
		if err != nil {
			return nil, dropped, &Error{Reason: ReasonPostprocessFail, Node: embedding.node.ID, Err: fmt.Errorf("object %d: %w", item.ID, err)}
		}
		vector, ok := decoded.(*decode.VectorResult)
		if !ok {
			return nil, dropped, &Error{Reason: ReasonPostprocessFail, Node: embedding.node.ID, Err: fmt.Errorf("decoder returned %T, want VectorResult", decoded)}
		}
		if len(vector.Vector) != 128 {
			return nil, dropped, &Error{Reason: ReasonPostprocessFail, Node: embedding.node.ID, Err: fmt.Errorf("SFace embedding must have 128 dimensions, got %d", len(vector.Vector))}
		}
		for _, value := range vector.Vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, dropped, &Error{Reason: ReasonPostprocessFail, Node: embedding.node.ID, Err: fmt.Errorf("SFace embedding contains a non-finite value")}
			}
		}
		item.Embedding = append([]float32(nil), vector.Vector...)
	}
	return items, dropped, nil
}

func releaseNode(node *resolvedNode) {
	if node != nil && node.lease != nil {
		node.lease.Release()
		node.lease = nil
	}
}

// Object is the bounded, source-frame face fact emitted to local consumers.
type Object struct {
	ID        int64             `json:"id"`
	ParentID  int64             `json:"parentId"`
	Type      string            `json:"type"`
	Box       [4]float32        `json:"box"`
	Score     float32           `json:"score"`
	Keypoints []decode.Keypoint `json:"keypoints"`
	Embedding []float32         `json:"embedding"`
}

// Result is one frame's PipelineJSON document. Objects is always a non-nil
// slice, including healthy empty frames.
type Result struct {
	Schema         string   `json:"schema"`
	PipelineID     string   `json:"pipelineId"`
	FrameID        uint64   `json:"frameId"`
	Objects        []Object `json:"objects"`
	DroppedObjects int      `json:"droppedObjects,omitempty"`

	maxEventBytes int
}

// Family implements decode.Result.
func (r *Result) Family() string { return "pipeline" }

// JSON implements decode.Result.
func (r *Result) JSON() any { return r }

// Draw is intentionally a no-op; pipeline v1 does not add a second renderer.
func (r *Result) Draw(*image.RGBA, []string) {}

// Readings emits exactly one bounded structured business reading. If the byte
// cap requires further truncation, lowest-scoring tail objects are dropped and
// accounted for rather than emitting invalid JSON or an unbounded event.
func (r *Result) Readings(capBytes int) ([]decode.Reading, error) {
	if r.maxEventBytes > 0 && (capBytes <= 0 || r.maxEventBytes < capBytes) {
		capBytes = r.maxEventBytes
	}
	candidate := *r
	candidate.Objects = append([]Object(nil), r.Objects...)
	if candidate.Objects == nil {
		candidate.Objects = []Object{}
	}
	for {
		payload, err := json.Marshal(&candidate)
		if err != nil {
			return nil, err
		}
		if capBytes <= 0 || len(payload) <= capBytes {
			return []decode.Reading{{Name: "PipelineJSON", Type: "string", Value: string(payload)}}, nil
		}
		if len(candidate.Objects) == 0 {
			return nil, fmt.Errorf("pipeline empty event %d bytes exceeds cap %d", len(payload), capBytes)
		}
		candidate.Objects = candidate.Objects[:len(candidate.Objects)-1]
		candidate.DroppedObjects++
	}
}
