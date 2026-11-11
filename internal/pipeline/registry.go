// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package pipeline owns NI-local, versioned linear inference pipelines. The
// registry deliberately accepts only the fixed source -> detector -> ROI
// transform -> embedding -> emit vocabulary implemented by Executor; it is not
// a general DAG or scripting environment.
package pipeline

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"

	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
)

const (
	// pipelineStoreRelPath is below ModelRoot, alongside other NI runtime state.
	pipelineStoreRelPath = "runtime/pipelines/pipelines.json"
)

// Definition is one immutable pipeline version. Jobs reference PipelineID and
// resolve the greatest accepted Version each time the graph is (re)resolved.
type Definition struct {
	PipelineID  string      `json:"pipelineId" yaml:"pipelineId"`
	Version     string      `json:"version" yaml:"version"`
	Description string      `json:"description,omitempty" yaml:"description,omitempty"`
	Nodes       []Node      `json:"nodes" yaml:"nodes"`
	Edges       [][2]string `json:"edges" yaml:"edges"`
	Limits      Limits      `json:"limits,omitempty" yaml:"limits,omitempty"`
}

// Node is one member of the closed linear vocabulary.
type Node struct {
	ID       string         `json:"id" yaml:"id"`
	Kind     string         `json:"kind" yaml:"kind"`
	Model    ModelReference `json:"model,omitempty" yaml:"model,omitempty"`
	Inputs   []InputBinding `json:"inputs,omitempty" yaml:"inputs,omitempty"`
	Operator string         `json:"operator,omitempty" yaml:"operator,omitempty"`
	// Mode is "five-point" for similarity landmark alignment or "roi" for a
	// plain ROI crop. Alignment geometry itself comes from the consumer
	// model's manifest, never from NL or from a model filename.
	Mode string `json:"mode,omitempty" yaml:"mode,omitempty"`
}

// ModelReference selects a catalog package for an infer node. Empty version
// means the newest installed package, using modelcatalog's normal resolution.
type ModelReference struct {
	ModelID      string `json:"modelId" yaml:"modelId"`
	ModelVersion string `json:"modelVersion,omitempty" yaml:"modelVersion,omitempty"`
	TargetDevice string `json:"targetDevice,omitempty" yaml:"targetDevice,omitempty"`
}

// InputBinding describes the only two v1 tensor sources. Name is an optional
// assertion; the descriptor's declared input name always constructs the wire
// request, so portable built-ins need not duplicate vendor tensor names.
type InputBinding struct {
	Name string `json:"name,omitempty" yaml:"name,omitempty"`
	From string `json:"from" yaml:"from"`
}

// Limits bounds work and emitted bytes for every frame.
type Limits struct {
	// MaxObjects keeps the highest-scoring detections and drops the rest before
	// the embedding stage. Overflow is reported as droppedObjects.
	MaxObjects int `json:"maxObjects" yaml:"maxObjects"`
	// MaxEventBytes is a per-pipeline cap, further bounded by the service-wide
	// JSONPayloadCap. If necessary, lowest-scoring objects are dropped from the
	// encoded event until it fits.
	MaxEventBytes int `json:"maxEventBytes" yaml:"maxEventBytes"`
}

// RegistryConfig selects the A-channel YAML and the persistent B-channel file.
type RegistryConfig struct {
	ModelRoot     string
	ProvisionFile string
}

// Registry holds the currently accepted full table. A successful PUT persists
// the complete replacement before atomically swapping the in-memory table.
// History retains content hashes for deleted versions so remove-and-re-add
// cannot bypass immutability, including across service restarts.
type Registry struct {
	storePath string

	mu      sync.RWMutex
	entries map[string]Definition
	history map[string]string
	subs    map[chan struct{}]struct{}

	replaceMu sync.Mutex
}

// storedRegistry is the private on-disk format. PipelinesJSON remains a plain
// Definition array; history is an implementation detail of crash persistence.
type storedRegistry struct {
	Pipelines []Definition      `json:"pipelines"`
	History   map[string]string `json:"history"`
}

// NewRegistry starts from an empty table, loads the opt-in provisioned
// definitions when ProvisionFile is set, then overlays a persisted
// full-table replacement when one exists. The platform binary ships no
// business pipeline: application graphs arrive through the provisioned YAML
// (samples under samples/) or PipelinesJSON PUT. Missing YAML is allowed and
// yields an empty registry; invalid existing state fails service
// initialization rather than partially loading a table.
func NewRegistry(cfg RegistryConfig) (*Registry, error) {
	if strings.TrimSpace(cfg.ModelRoot) == "" {
		return nil, fmt.Errorf("pipeline registry model root is required")
	}
	r := &Registry{
		storePath: filepath.Join(filepath.Clean(cfg.ModelRoot), filepath.FromSlash(pipelineStoreRelPath)),
		entries:   map[string]Definition{},
		history:   map[string]string{},
		subs:      map[chan struct{}]struct{}{},
	}
	if cfg.ProvisionFile != "" {
		b, err := os.ReadFile(cfg.ProvisionFile)
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("read provisioned pipelines: %w", err)
		}
		if err == nil {
			var table struct {
				Pipelines []Definition `yaml:"pipelines"`
			}
			if err := yaml.Unmarshal(b, &table); err != nil {
				return nil, fmt.Errorf("parse provisioned pipelines %q: %w", cfg.ProvisionFile, err)
			}
			provisioned, err := validateTable(table.Pipelines)
			if err != nil {
				return nil, fmt.Errorf("provisioned pipelines: %w", err)
			}
			for key, def := range provisioned {
				if old, exists := r.entries[key]; exists && !definitionsEqual(old, def) {
					return nil, fmt.Errorf("provisioned pipeline %s/%s redefines an accepted immutable version", def.PipelineID, def.Version)
				}
				r.entries[key] = def
				r.history[key] = definitionHash(def)
			}
		}
	}
	b, err := os.ReadFile(r.storePath)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("read persisted pipelines: %w", err)
	}
	if err == nil {
		state, decodeErr := decodeStoredRegistryJSON(b)
		if decodeErr != nil {
			return nil, fmt.Errorf("parse persisted pipelines: %w", decodeErr)
		}
		persisted, err := validateTable(state.Pipelines)
		if err != nil {
			return nil, fmt.Errorf("persisted pipelines: %w", err)
		}
		for key, hash := range state.History {
			decoded, decodeErr := hex.DecodeString(hash)
			if decodeErr != nil || len(decoded) != sha256.Size {
				return nil, fmt.Errorf("persisted pipeline history %q has an invalid hash", key)
			}
			if current, exists := r.history[key]; exists && current != hash {
				return nil, fmt.Errorf("persisted pipeline history conflicts with an immutable provisioned definition")
			}
			r.history[key] = hash
		}
		for key, def := range persisted {
			hash := definitionHash(def)
			if accepted, exists := r.history[key]; exists && accepted != hash {
				return nil, fmt.Errorf("persisted pipeline %s/%s changes an immutable version", def.PipelineID, def.Version)
			}
			r.history[key] = hash
		}
		r.entries = persisted
	}
	return r, nil
}

// Lookup resolves the latest accepted version of id.
func (r *Registry) Lookup(id string) (Definition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var best Definition
	found := false
	for _, def := range r.entries {
		if def.PipelineID != id {
			continue
		}
		if !found || compareVersionIdentity(def.Version, best.Version) > 0 {
			best, found = def, true
		}
	}
	return cloneDefinition(best), found
}

// Snapshot returns the complete accepted table in stable id/version order.
func (r *Registry) Snapshot() []Definition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Definition, 0, len(r.entries))
	for _, def := range r.entries {
		out = append(out, cloneDefinition(def))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].PipelineID != out[j].PipelineID {
			return out[i].PipelineID < out[j].PipelineID
		}
		return compareVersionIdentity(out[i].Version, out[j].Version) < 0
	})
	return out
}

// Replace validates and persists a complete replacement. Accepted id+version
// entries may be retained byte-equivalently or removed, but never modified,
// including when a later replacement re-adds a previously deleted version.
// No map or subscriber changes occur until the atomic rename succeeds.
func (r *Registry) Replace(records []Definition) error {
	r.replaceMu.Lock()
	defer r.replaceMu.Unlock()
	next, err := validateTable(records)
	if err != nil {
		return err
	}
	r.mu.RLock()
	nextHistory := make(map[string]string, len(r.history)+len(next))
	for key, hash := range r.history {
		nextHistory[key] = hash
	}
	for key, def := range next {
		hash := definitionHash(def)
		if accepted, exists := nextHistory[key]; exists && accepted != hash {
			r.mu.RUnlock()
			return fmt.Errorf("pipeline %s/%s is immutable", def.PipelineID, def.Version)
		}
		nextHistory[key] = hash
	}
	r.mu.RUnlock()

	canonical := make([]Definition, 0, len(next))
	for _, def := range next {
		canonical = append(canonical, def)
	}
	sort.Slice(canonical, func(i, j int) bool {
		if canonical[i].PipelineID != canonical[j].PipelineID {
			return canonical[i].PipelineID < canonical[j].PipelineID
		}
		return compareVersionIdentity(canonical[i].Version, canonical[j].Version) < 0
	})
	b, err := json.MarshalIndent(storedRegistry{Pipelines: canonical, History: nextHistory}, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.MkdirAll(filepath.Dir(r.storePath), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(r.storePath), ".pipelines-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, r.storePath); err != nil {
		return err
	}

	r.mu.Lock()
	r.entries = next
	r.history = nextHistory
	for ch := range r.subs {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
	r.mu.Unlock()
	return nil
}

// Subscribe returns a coalescing update notification channel. Definitions are
// fetched through Lookup, so consumers never share mutable registry slices.
func (r *Registry) Subscribe() (<-chan struct{}, func()) {
	ch := make(chan struct{}, 1)
	r.mu.Lock()
	r.subs[ch] = struct{}{}
	r.mu.Unlock()
	return ch, func() {
		r.mu.Lock()
		delete(r.subs, ch)
		r.mu.Unlock()
	}
}

func validateTable(records []Definition) (map[string]Definition, error) {
	next := make(map[string]Definition, len(records))
	for i, raw := range records {
		def := normalizeDefinition(raw)
		if err := ValidateDefinition(def); err != nil {
			return nil, fmt.Errorf("pipeline[%d] %q/%q: %w", i, raw.PipelineID, raw.Version, err)
		}
		key := definitionKey(def.PipelineID, def.Version)
		if _, duplicate := next[key]; duplicate {
			return nil, fmt.Errorf("duplicate pipeline %s/%s", def.PipelineID, def.Version)
		}
		next[key] = cloneDefinition(def)
	}
	return next, nil
}

// ValidateDefinition enforces exactly the five-node chain Executor implements.
func ValidateDefinition(def Definition) error {
	if !model.IsSafeName(def.PipelineID) {
		return fmt.Errorf("pipelineId must be a safe name, got %q", def.PipelineID)
	}
	if !model.IsSafeName(def.Version) {
		return fmt.Errorf("version must be a safe name, got %q", def.Version)
	}
	if len(def.Description) > 512 {
		return fmt.Errorf("description exceeds 512 bytes")
	}
	if len(def.Nodes) != 5 || len(def.Edges) != 4 {
		return fmt.Errorf("minimal pipeline requires five nodes and four edges")
	}
	kinds := []string{"source", "infer", "transform", "infer", "emit"}
	seen := map[string]struct{}{}
	for i, node := range def.Nodes {
		if !model.IsSafeName(node.ID) {
			return fmt.Errorf("node[%d] has unsafe id %q", i, node.ID)
		}
		if _, duplicate := seen[node.ID]; duplicate {
			return fmt.Errorf("duplicate node id %q", node.ID)
		}
		seen[node.ID] = struct{}{}
		if node.Kind != kinds[i] {
			return fmt.Errorf("node[%d] kind must be %q, got %q", i, kinds[i], node.Kind)
		}
		if i > 0 && (def.Edges[i-1][0] != def.Nodes[i-1].ID || def.Edges[i-1][1] != node.ID) {
			return fmt.Errorf("edges must connect the declared linear node order")
		}
		switch node.Kind {
		case "infer":
			if !model.IsSafeName(node.Model.ModelID) {
				return fmt.Errorf("infer node %q has invalid modelId %q", node.ID, node.Model.ModelID)
			}
			if node.Model.ModelVersion != "" && !model.IsSafeName(node.Model.ModelVersion) {
				return fmt.Errorf("infer node %q has invalid modelVersion %q", node.ID, node.Model.ModelVersion)
			}
			if node.Model.TargetDevice != "" {
				if _, err := modelserver.NormalizeTargetDevice(node.Model.TargetDevice); err != nil {
					return fmt.Errorf("infer node %q: %w", node.ID, err)
				}
			}
			if len(node.Inputs) != 1 {
				return fmt.Errorf("infer node %q requires exactly one input", node.ID)
			}
			wantFrom := "frame"
			if i == 3 {
				wantFrom = "crops"
			}
			if node.Inputs[0].From != wantFrom {
				return fmt.Errorf("infer node %q input.from must be %q", node.ID, wantFrom)
			}
			if node.Inputs[0].Name != "" && !model.IsSafeName(node.Inputs[0].Name) {
				return fmt.Errorf("infer node %q has invalid input name %q", node.ID, node.Inputs[0].Name)
			}
		case "transform":
			if node.Operator != "roi-align-crop" {
				return fmt.Errorf("transform node %q supports only roi-align-crop", node.ID)
			}
			if node.Mode != "five-point" && node.Mode != "roi" {
				return fmt.Errorf("transform node %q mode must be five-point or roi", node.ID)
			}
		}
	}
	if def.Limits.MaxObjects < 1 || def.Limits.MaxObjects > 64 {
		return fmt.Errorf("limits.maxObjects must be in 1..64")
	}
	if def.Limits.MaxEventBytes < 1024 || def.Limits.MaxEventBytes > 1<<20 {
		return fmt.Errorf("limits.maxEventBytes must be in 1024..1048576")
	}
	return nil
}

func normalizeDefinition(def Definition) Definition {
	def = cloneDefinition(def)
	def.PipelineID = strings.TrimSpace(def.PipelineID)
	def.Version = strings.TrimSpace(def.Version)
	for i := range def.Nodes {
		def.Nodes[i].Model.ModelID = strings.TrimSpace(def.Nodes[i].Model.ModelID)
		def.Nodes[i].Model.ModelVersion = strings.TrimSpace(def.Nodes[i].Model.ModelVersion)
		target := strings.TrimSpace(def.Nodes[i].Model.TargetDevice)
		def.Nodes[i].Model.TargetDevice = target
		if target != "" {
			if normalized, err := modelserver.NormalizeTargetDevice(target); err == nil {
				def.Nodes[i].Model.TargetDevice = normalized
			}
		}
		if def.Nodes[i].Kind == "transform" && def.Nodes[i].Mode == "" {
			def.Nodes[i].Mode = "five-point"
		}
	}
	if def.Limits.MaxObjects == 0 {
		def.Limits.MaxObjects = 8
	}
	if def.Limits.MaxEventBytes == 0 {
		def.Limits.MaxEventBytes = 64 * 1024
	}
	return def
}

func definitionsEqual(a, b Definition) bool {
	ab, aerr := json.Marshal(normalizeDefinition(a))
	bb, berr := json.Marshal(normalizeDefinition(b))
	return aerr == nil && berr == nil && string(ab) == string(bb)
}

func definitionHash(def Definition) string {
	canonical, _ := json.Marshal(normalizeDefinition(def))
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

func definitionKey(id, version string) string { return id + "\x00" + version }

func cloneDefinition(def Definition) Definition {
	cp := def
	cp.Nodes = append([]Node(nil), def.Nodes...)
	for i := range cp.Nodes {
		cp.Nodes[i].Inputs = append([]InputBinding(nil), def.Nodes[i].Inputs...)
	}
	cp.Edges = append([][2]string(nil), def.Edges...)
	return cp
}

// compareVersionIdentity adds a lexical tie-break for numerically equivalent
// aliases such as 1.01.0 and 1.1.0, making map iteration order irrelevant.
func compareVersionIdentity(a, b string) int {
	if cmp := compareVersions(a, b); cmp != 0 {
		return cmp
	}
	return strings.Compare(a, b)
}

// compareVersions defines a strict total preorder over dotted safe strings.
// Numeric components compare at arbitrary precision, numeric components sort
// before non-numeric components, and fewer components sort first. Numerically
// equivalent aliases remain equal here so compareVersionIdentity can apply the
// original-string tie-break without introducing map-order dependence.
func compareVersions(a, b string) int {
	as, bs := strings.Split(a, "."), strings.Split(b, ".")
	shared := len(as)
	if len(bs) < shared {
		shared = len(bs)
	}
	for i := 0; i < shared; i++ {
		ac, bc := as[i], bs[i]
		if ac == "" || bc == "" {
			if ac != bc {
				if ac == "" {
					return -1
				}
				return 1
			}
			continue
		}
		aNumeric, bNumeric := asciiDigits(ac), asciiDigits(bc)
		switch {
		case aNumeric && bNumeric:
			an, bn := significantDigits(ac), significantDigits(bc)
			if len(an) != len(bn) {
				if len(an) < len(bn) {
					return -1
				}
				return 1
			}
			if cmp := strings.Compare(an, bn); cmp != 0 {
				return cmp
			}
		case aNumeric != bNumeric:
			if aNumeric {
				return -1
			}
			return 1
		default:
			if cmp := strings.Compare(ac, bc); cmp != 0 {
				return cmp
			}
		}
	}
	if len(as) < len(bs) {
		return -1
	}
	if len(as) > len(bs) {
		return 1
	}
	return 0
}

func asciiDigits(value string) bool {
	if value == "" {
		return false
	}
	for i := 0; i < len(value); i++ {
		if value[i] < '0' || value[i] > '9' {
			return false
		}
	}
	return true
}

func significantDigits(value string) string {
	trimmed := strings.TrimLeft(value, "0")
	if trimmed == "" {
		return "0"
	}
	return trimmed
}
