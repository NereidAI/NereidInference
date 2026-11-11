// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package model defines the in-memory representation of a Nereid model
// package descriptor (nereid-model.json, schema "nereid-model/1") plus the
// labels and auxiliary files shipped alongside it.
//
// The descriptor is the single source of truth for request construction,
// preprocessing and tensor decoding; generic code must never branch on task
// family, vendor or model name. See docs/2026-10-25-edge-inference-
// platform-contract.md section 4.3 and docs/2026-10-26-inference-ds-
// implementation.md section 7.
package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// SchemaID is the accepted manifest schema value.
const SchemaID = "nereid-model/1"

// Known artifact formats produced by NereidForge. Runtime adapters decide
// which formats each server can serve.
const (
	ArtifactFormatONNX       = "onnx"
	ArtifactFormatOpenVINOIR = "openvino-ir"
)

// Artifact precisions registered in the NF/NI package contract (contract
// v0.3 §4.6). An empty value means the packager did not declare one.
const (
	PrecisionFP32 = "FP32"
	PrecisionFP16 = "FP16"
	PrecisionINT8 = "INT8"
)

// labelIndexedKinds are decoder kinds whose output class indices name labels;
// their packages must ship a labels file (contract v0.3 §9 R2).
var labelIndexedKinds = map[string]bool{
	"detection.anchors":         true,
	"detection.rows":            true,
	"classification.probs":      true,
	"classification.multilabel": true,
	"segmentation.semantic":     true,
	"segmentation.instance":     true,
	"keypoints":                 true,
}

// safeNameRe permits model IDs/versions/revisions to be used as single path
// components: no separators, traversal, NUL, shell metacharacters, or spaces.
var safeNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$`)

// IsSafeName reports whether value is safe to use as one path component.
func IsSafeName(value string) bool { return safeNameRe.MatchString(value) }

// State values stored in the catalog ledger.
const (
	StateExtracting = "EXTRACTING"
	StateValidating = "VALIDATING"
	StateReady      = "READY"
	StateFailed     = "FAILED"
	StateIgnored    = "IGNORED"
	StateRemoved    = "REMOVED"
)

// Manifest is the parsed package descriptor.
type Manifest struct {
	Schema   string `json:"schema"`
	Model    ModelSection
	Artifact ArtifactSection
	Runtime  RuntimeSection
	Hardware HardwareSection
	IO       IOSection
	// Decoder configures the generic output decoder; its Kind indexes the
	// decoder registry while the remaining fields are decoded per kind.
	Decoder     json.RawMessage `json:"decoder,omitempty"`
	Postprocess json.RawMessage `json:"postprocess,omitempty"`
}

// ModelSection identifies the model and its task.
type ModelSection struct {
	ID         string `json:"id"`
	Version    string `json:"version"`
	Revision   string `json:"revision"`
	Task       string `json:"task"`
	Family     string `json:"family"`
	LabelsFile string `json:"labels_file"`
}

// ArtifactSection lists payload files for integrity checks.
type ArtifactSection struct {
	Format    string      `json:"format"`
	Precision string      `json:"precision"`
	Files     []FileEntry `json:"files"`
	// SourcePath records the tarball location; it is catalog runtime state
	// (not serialized) and drives ESS/inbox delete linkage.
	SourcePath string `json:"-"`
}

// FileEntry is one payload file with size and SHA-256 digest.
type FileEntry struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

// RuntimeSection selects backend and the single server-ready repository in
// this package. Deployment target (CPU/GPU/NPU) is a binding-time decision and
// is intentionally not part of the immutable descriptor; hardware-specific
// artifacts are delivered as separate distribution packages.
type RuntimeSection struct {
	Backend        string `json:"backend"`
	ModelName      string `json:"model_name"`
	RepositoryPath string `json:"repository_path,omitempty"`
	Server         struct {
		Family     string `json:"family"`
		Image      string `json:"image"`
		MinVersion string `json:"min_version"`
	} `json:"server"`
	// ServedBy optionally maps this package to multiple server-ready payload
	// slots for different managed families (NF multi-vendor packaging). When
	// empty, adapters fall back to RepositoryPath. Unknown families are
	// rejected at validation; see modelserver family tokens.
	ServedBy []ServedByEntry `json:"served_by,omitempty"`
}

// ServedByEntry is one payload slot selectable by family and binding target.
// The payload path is relative to the package root and must stay inside it.
type ServedByEntry struct {
	// Family is the modelserver family token (e.g. "ovms", "triton").
	Family string `json:"family"`
	// BackendHint names the server backend when one family has several
	// (e.g. Triton tensorrt vs onnxruntime); adapters validate it against an
	// allowlist before rendering server configuration.
	BackendHint string `json:"backend_hint,omitempty"`
	// Payload is the package-relative server-ready repository directory.
	Payload string `json:"payload"`
	// Targets optionally restricts the slot to binding target devices; empty
	// means the slot matches every target for its family.
	Targets []string `json:"targets,omitempty"`
	// Requires carries SKU/SDK compatibility metadata (compute capability,
	// engine/driver floors); it is informational for this service in v1 and
	// never interpolated into commands.
	Requires map[string]string `json:"requires,omitempty"`
}

// HardwareSection declares hardware compatibility; informational for the DS.
type HardwareSection struct {
	Vendor      string            `json:"vendor"`
	Device      string            `json:"device"`
	Constraints map[string]string `json:"constraints"`
}

// IOSection declares model inputs and outputs.
type IOSection struct {
	Inputs  []IOSpec `json:"inputs"`
	Outputs []IOSpec `json:"outputs"`
}

// IOSpec describes one tensor endpoint.
type IOSpec struct {
	Name       string          `json:"name"`
	Shape      []int64         `json:"shape"`
	Layout     string          `json:"layout"`
	DType      string          `json:"dtype"`
	Kind       string          `json:"kind,omitempty"`
	ColorOrder string          `json:"color_order,omitempty"`
	Role       string          `json:"role,omitempty"`
	Preprocess *PreprocessSpec `json:"preprocess,omitempty"`
}

// PreprocessSpec drives the pure-Go image preprocessing stage.
type PreprocessSpec struct {
	Resize   string    `json:"resize"` // letterbox|stretch|crop
	Scale    string    `json:"scale"`  // expression such as "1/255"
	Mean     []float32 `json:"mean"`
	Std      []float32 `json:"std"`
	PadValue float32   `json:"pad_value"`
	// Alignment optionally declares the consumer model's five-point destination
	// template in input-pixel coordinates. roi-align-crop uses this geometry;
	// graph definitions and NL never duplicate ArcFace/SFace constants.
	Alignment *AlignmentSpec `json:"alignment,omitempty"`
}

// AlignmentSpec is a descriptor-owned five-landmark destination template.
type AlignmentSpec struct {
	Template [][2]float32 `json:"template"`
}

// Load reads and validates a manifest from an unpacked active directory.
func Load(dir string) (*Manifest, []string, error) {
	p := filepath.Join(dir, "nereid-model.json")
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, nil, fmt.Errorf("read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, nil, fmt.Errorf("parse manifest: %w", err)
	}
	if err := m.validate(); err != nil {
		return nil, nil, err
	}
	labels, err := m.loadLabels(dir)
	if err != nil {
		return nil, nil, err
	}
	return &m, labels, nil
}

func (m Manifest) validate() error {
	if m.Schema != SchemaID {
		return fmt.Errorf("unsupported manifest schema %q", m.Schema)
	}
	if m.Model.ID == "" || m.Model.Version == "" || m.Model.Task == "" {
		return fmt.Errorf("manifest model id/version/task are required")
	}
	if !IsSafeName(m.Model.ID) {
		return fmt.Errorf("unsafe model id %q", m.Model.ID)
	}
	if !IsSafeName(m.Model.Version) {
		return fmt.Errorf("unsafe model version %q", m.Model.Version)
	}
	if m.Model.Revision != "" && !IsSafeName(m.Model.Revision) {
		return fmt.Errorf("unsafe model revision %q", m.Model.Revision)
	}
	if m.Runtime.Backend == "" || m.Runtime.ModelName == "" {
		return fmt.Errorf("manifest runtime backend/model_name are required")
	}
	if !IsSafeName(m.Runtime.ModelName) {
		return fmt.Errorf("unsafe runtime model_name %q", m.Runtime.ModelName)
	}
	switch m.Runtime.Backend {
	case "kserve", "openai":
	default:
		return fmt.Errorf("unsupported runtime backend %q", m.Runtime.Backend)
	}
	if m.Runtime.Backend == "kserve" {
		if m.Artifact.Format == "" {
			return fmt.Errorf("manifest artifact format is required for kserve packages")
		}
		switch m.Artifact.Precision {
		case "", PrecisionFP32, PrecisionFP16, PrecisionINT8:
		default:
			return fmt.Errorf("unsupported artifact precision %q", m.Artifact.Precision)
		}
		if m.Runtime.Server.Family == "" {
			return fmt.Errorf("manifest runtime.server.family is required for kserve packages")
		}
		switch m.Runtime.Server.Family {
		case "ovms", "triton", "external":
		default:
			return fmt.Errorf("unsupported runtime.server.family %q", m.Runtime.Server.Family)
		}
		if len(m.IO.Inputs) == 0 {
			return fmt.Errorf("manifest io.inputs must declare at least one tensor input")
		}
		if err := validateVisionInput(m.IO.Inputs[0]); err != nil {
			return err
		}
		for i := range m.IO.Outputs {
			if err := validateOutput(m.IO.Outputs[i]); err != nil {
				return err
			}
		}
		if err := m.validateDecoder(); err != nil {
			return err
		}
		if m.Runtime.RepositoryPath != "" {
			if _, err := safeRelativePath(m.Runtime.RepositoryPath); err != nil {
				return fmt.Errorf("runtime.repository_path: %w", err)
			}
		}
		if err := m.validateServedBy(); err != nil {
			return err
		}
	}
	if m.Model.LabelsFile != "" {
		if _, err := safeRelativePath(m.Model.LabelsFile); err != nil {
			return fmt.Errorf("labels_file: %w", err)
		}
	}
	return nil
}

// validateServedBy keeps the optional multi-family payload map tamper-proof:
// families come from the registry vocabulary, every payload is a safe
// package-relative path, and target/requirement values stay short tokens.
func (m Manifest) validateServedBy() error {
	seen := map[string]struct{}{}
	for i, slot := range m.Runtime.ServedBy {
		switch slot.Family {
		case "ovms", "triton", "external":
		default:
			return fmt.Errorf("runtime.served_by[%d]: unsupported family %q", i, slot.Family)
		}
		if strings.TrimSpace(slot.Payload) == "" {
			return fmt.Errorf("runtime.served_by[%d]: payload is required", i)
		}
		if _, err := safeRelativePath(slot.Payload); err != nil {
			return fmt.Errorf("runtime.served_by[%d].payload: %w", i, err)
		}
		if slot.BackendHint != "" && !safeNameRe.MatchString(slot.BackendHint) {
			return fmt.Errorf("runtime.served_by[%d].backend_hint must be a safe token, got %q", i, slot.BackendHint)
		}
		for _, target := range slot.Targets {
			if len(target) > 64 || !safeNameRe.MatchString(target) {
				return fmt.Errorf("runtime.served_by[%d] has invalid target %q", i, target)
			}
		}
		key := slot.Family + "\x00" + strings.ToUpper(strings.TrimSpace(slot.BackendHint))
		if _, dup := seen[key]; dup {
			return fmt.Errorf("runtime.served_by[%d]: duplicate family/backend_hint slot %q", i, key)
		}
		seen[key] = struct{}{}
		for k, v := range slot.Requires {
			if len(k) > 64 || len(v) > 256 {
				return fmt.Errorf("runtime.served_by[%d].requires entry %q is too long", i, k)
			}
		}
	}
	return nil
}

func validateVisionInput(in IOSpec) error {
	if in.Name == "" || in.DType != "FP32" || in.Layout != "NCHW" {
		return fmt.Errorf("vision input must be named FP32 NCHW, got name=%q dtype=%q layout=%q", in.Name, in.DType, in.Layout)
	}
	if len(in.Shape) != 4 || in.Shape[0] != 1 || in.Shape[1] != 3 || in.Shape[2] <= 0 || in.Shape[3] <= 0 {
		return fmt.Errorf("vision input shape must be [1,3,H,W], got %v", in.Shape)
	}
	const maxSpatial = 4096
	if in.Shape[2] > maxSpatial || in.Shape[3] > maxSpatial {
		return fmt.Errorf("vision input dimensions exceed %d: %v", maxSpatial, in.Shape)
	}
	if in.ColorOrder != "" && in.ColorOrder != "RGB" && in.ColorOrder != "BGR" {
		return fmt.Errorf("unsupported color_order %q", in.ColorOrder)
	}
	if pp := in.Preprocess; pp != nil {
		switch pp.Resize {
		case "", "stretch", "letterbox", "crop":
		default:
			return fmt.Errorf("unsupported preprocess resize %q", pp.Resize)
		}
		if len(pp.Mean) != 0 && len(pp.Mean) != 1 && len(pp.Mean) != 3 {
			return fmt.Errorf("preprocess mean must have 1 or 3 values")
		}
		if len(pp.Std) != 0 && len(pp.Std) != 1 && len(pp.Std) != 3 {
			return fmt.Errorf("preprocess std must have 1 or 3 values")
		}
		if pp.PadValue < 0 || pp.PadValue > 255 {
			return fmt.Errorf("preprocess pad_value must be in 0..255")
		}
		if pp.Alignment != nil {
			if len(pp.Alignment.Template) != 5 {
				return fmt.Errorf("preprocess alignment template must contain exactly five points")
			}
			for i, point := range pp.Alignment.Template {
				x, y := point[0], point[1]
				if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) || math.IsInf(float64(x), 0) || math.IsInf(float64(y), 0) {
					return fmt.Errorf("preprocess alignment template point %d is not finite", i)
				}
				if x < 0 || y < 0 || x > float32(in.Shape[3]-1) || y > float32(in.Shape[2]-1) {
					return fmt.Errorf("preprocess alignment template point %d lies outside input bounds", i)
				}
			}
		}
	}
	return nil
}

// validateOutput checks one declared output tensor. Dynamic dimensions use
// -1 (allowed for non-anchors kinds); zero or other negatives are invalid.
func validateOutput(out IOSpec) error {
	if out.Name == "" {
		return fmt.Errorf("io.outputs entry must declare a name")
	}
	for _, d := range out.Shape {
		if d == 0 || d < -1 {
			return fmt.Errorf("output %q has invalid shape %v", out.Name, out.Shape)
		}
	}
	return nil
}

// validateDecoder enforces contract v0.3 decoder-level rules: class-indexed
// kinds require labels, referenced output names must resolve, and the anchors
// kind requires a static [1,4+nc,A] output (NPU static-shape constraint).
func (m Manifest) validateDecoder() error {
	if len(m.Decoder) == 0 {
		return nil
	}
	kind, err := DecoderKind(m.Decoder)
	if err != nil {
		return err
	}
	if labelIndexedKinds[kind] && m.Model.LabelsFile == "" {
		return fmt.Errorf("decoder kind %q requires model.labels_file", kind)
	}
	var refs struct {
		Output      string `json:"output"`
		Predictions string `json:"predictions"`
		MaskProtos  string `json:"mask_protos"`
	}
	if err := json.Unmarshal(m.Decoder, &refs); err != nil {
		return fmt.Errorf("parse decoder descriptor: %w", err)
	}
	hasOutput := func(name string) bool {
		for _, out := range m.IO.Outputs {
			if out.Name == name || out.Role == name {
				return true
			}
		}
		return false
	}
	for _, ref := range []string{refs.Output, refs.Predictions, refs.MaskProtos} {
		if ref != "" && !hasOutput(ref) {
			return fmt.Errorf("decoder references undeclared output %q", ref)
		}
	}
	if kind == "detection.anchors" {
		name := refs.Output
		var spec *IOSpec
		for i := range m.IO.Outputs {
			if name == "" || m.IO.Outputs[i].Name == name || m.IO.Outputs[i].Role == name {
				spec = &m.IO.Outputs[i]
				break
			}
		}
		if spec == nil {
			return fmt.Errorf("detection.anchors output %q is not declared", name)
		}
		if len(spec.Shape) != 3 || spec.Shape[0] != 1 || spec.Shape[1] < 5 || spec.Shape[2] <= 0 {
			return fmt.Errorf("detection.anchors output must be static [1,4+nc,A] with A>0, got %v", spec.Shape)
		}
	}
	if kind == "detection.yunet" {
		if err := m.validateYuNetDecoder(hasOutput); err != nil {
			return err
		}
	}
	return nil
}

// validateYuNetDecoder checks all twelve explicit output references and the
// static parts of their anchor-major shapes at package admission. Runtime
// anchor counts and tensor lengths are checked again against actual responses.
func (m Manifest) validateYuNetDecoder(hasOutput func(string) bool) error {
	var spec struct {
		Threshold  float32 `json:"threshold"`
		Activation string  `json:"activation"`
		Strides    []struct {
			Stride int    `json:"stride"`
			Cls    string `json:"cls"`
			Obj    string `json:"obj"`
			BBox   string `json:"bbox"`
			KPS    string `json:"kps"`
		} `json:"strides"`
	}
	if err := json.Unmarshal(m.Decoder, &spec); err != nil {
		return fmt.Errorf("parse detection.yunet descriptor: %w", err)
	}
	if spec.Threshold < 0 || spec.Threshold > 1 {
		return fmt.Errorf("detection.yunet threshold must be in 0..1")
	}
	if spec.Activation != "" && spec.Activation != "probability" && spec.Activation != "logit" {
		return fmt.Errorf("detection.yunet activation must be probability or logit, got %q", spec.Activation)
	}
	if len(spec.Strides) != 3 {
		return fmt.Errorf("detection.yunet requires strides 8, 16 and 32")
	}
	input := m.IO.Inputs[0]
	if input.Shape[2]%8 != 0 || input.Shape[3]%8 != 0 || input.Shape[2]%16 != 0 || input.Shape[3]%16 != 0 || input.Shape[2]%32 != 0 || input.Shape[3]%32 != 0 {
		return fmt.Errorf("detection.yunet input spatial dimensions must be divisible by 8, 16 and 32")
	}
	seenStrides := map[int]bool{}
	seenOutputs := map[string]bool{}
	for _, stride := range spec.Strides {
		if stride.Stride != 8 && stride.Stride != 16 && stride.Stride != 32 {
			return fmt.Errorf("detection.yunet has unsupported stride %d", stride.Stride)
		}
		if seenStrides[stride.Stride] {
			return fmt.Errorf("detection.yunet has duplicate stride %d", stride.Stride)
		}
		seenStrides[stride.Stride] = true
		for _, ref := range []struct {
			name     string
			channels int64
		}{{stride.Cls, 1}, {stride.Obj, 1}, {stride.BBox, 4}, {stride.KPS, 10}} {
			if ref.name == "" || !hasOutput(ref.name) {
				return fmt.Errorf("detection.yunet stride %d references undeclared output %q", stride.Stride, ref.name)
			}
			if seenOutputs[ref.name] {
				return fmt.Errorf("detection.yunet output %q is referenced more than once", ref.name)
			}
			seenOutputs[ref.name] = true
			for _, out := range m.IO.Outputs {
				if out.Name != ref.name && out.Role != ref.name {
					continue
				}
				if err := validateYuNetOutputShape(out, ref.channels); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func validateYuNetOutputShape(out IOSpec, channels int64) error {
	if len(out.Shape) < 2 || len(out.Shape) > 4 || out.Shape[len(out.Shape)-1] != channels {
		return fmt.Errorf("detection.yunet output %q must be [A,%d], [1,A,%d] or [1,H,W,%d], got %v", out.Name, channels, channels, channels, out.Shape)
	}
	if len(out.Shape) >= 3 && out.Shape[0] != 1 {
		return fmt.Errorf("detection.yunet output %q must have batch 1, got %v", out.Name, out.Shape)
	}
	return nil
}

func safeRelativePath(path string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(path))
	if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes package directory", path)
	}
	return clean, nil
}

func (m Manifest) loadLabels(dir string) ([]string, error) {
	if m.Model.LabelsFile == "" {
		return nil, nil
	}
	rel, err := safeRelativePath(m.Model.LabelsFile)
	if err != nil {
		return nil, err
	}
	p := filepath.Join(dir, rel)
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	lines := strings.Split(strings.TrimRight(string(b), "\r\n"), "\n")
	return lines, nil
}

// DecoderKind extracts the "kind" field from a raw decoder descriptor.
func DecoderKind(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", fmt.Errorf("model %s has no decoder descriptor", "")
	}
	var head struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return "", err
	}
	if head.Kind == "" {
		return "", fmt.Errorf("decoder descriptor has no kind")
	}
	return head.Kind, nil
}
