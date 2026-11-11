// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/edgexfoundry/device-inference/internal/model"
)

// ErrExternalServablesReadOnly is returned for every managed write to the
// merged provisioned/legacy-runtime health view. NI does not currently support
// managed external delivery; provisioned records can never be converted into
// runtime-owned records through a merged GET -> PUT cycle.
var ErrExternalServablesReadOnly = errors.New("config-invalid: merged external-servable status is read-only")

const (
	// externalFamily marks servers NI cannot administrate.
	externalFamily = FamilyExternal
	// externalDirRelPath is the DS-owned external registry directory below
	// ModelRoot; the runtime table persists across restarts.
	externalDirRelPath = "runtime/external"
	// externalStoreFile holds legacy runtime records for read-only restart
	// visibility. Managed writes are disabled in the current contract.
	externalStoreFile = "servables.json"
	// externalProbeInterval bounds readiness refresh latency.
	externalProbeInterval = 15 * time.Second
	// externalProbeTimeout bounds one OIP probe.
	externalProbeTimeout = 6 * time.Second
	// externalProbeConcurrency bounds parallel server probes.
	externalProbeConcurrency = 4
	// externalStableCount implements the three-failure/three-success flap
	// guard used by the manager inventory.
	externalStableCount = 3
	// externalProtocolKServeGRPC is the only v1 external wire protocol.
	externalProtocolKServeGRPC = "kserve-grpc"
)

// External reason codes mirror the job LastError vocabulary without importing
// the job package (job imports modelserver, so the dependency cannot reverse).
const (
	externalReasonConfigInvalid  = "config-invalid"
	externalReasonRuntimeUnreach = "runtime-unreachable"
	externalReasonModelNotReady  = "model-not-ready"
	externalReasonRuntimeError   = "runtime-error"
)

// ServableCapabilities declares what the external server operator exposes.
type ServableCapabilities struct {
	// Devices lists the server-visible device identifiers.
	Devices []string `json:"devices,omitempty" yaml:"devices,omitempty"`
	// DeviceSelectable is false for every v1 record: placement lives in the
	// remote server's own repository, so bindings must not send TargetDevice.
	DeviceSelectable bool `json:"deviceSelectable" yaml:"deviceSelectable"`
}

// ServableRecord is one address-book entry for an inference server NI cannot
// administrate. NI probes it but never sends load/unload. Job execution through
// ServableID is currently rejected at admission.
type ServableRecord struct {
	ServableID   string               `json:"servableId" yaml:"servableId"`
	Family       string               `json:"family" yaml:"family"`
	Protocol     string               `json:"protocol" yaml:"protocol"`
	Endpoint     string               `json:"endpoint" yaml:"endpoint"`
	ModelName    string               `json:"modelName" yaml:"modelName"`
	ModelVersion string               `json:"modelVersion,omitempty" yaml:"modelVersion,omitempty"`
	Capabilities ServableCapabilities `json:"capabilities,omitempty" yaml:"capabilities,omitempty"`
	// Descriptor carries the runtime/io/decoder/labels sections of a standard
	// nereid-model.json descriptor. It never carries payloads or checksums.
	Descriptor json.RawMessage `json:"descriptor,omitempty" yaml:"descriptor,omitempty"`
}

// ServableStatus is the northbound view: one record plus its latest probe
// state and the channel that provisioned it ("provisioned" or "runtime").
type ServableStatus struct {
	ServableRecord
	Channel string `json:"channel"`
	State   string `json:"state"`
	Reason  string `json:"reason,omitempty"`
}

// ExternalProber executes readiness checks against an external server. The
// production adapter wraps infer.Registry; tests supply a fake.
type ExternalProber interface {
	Probe(ctx context.Context, protocol, endpoint, modelName, modelVersion string) error
}

// ExternalConfig configures the unmanaged-server registry.
type ExternalConfig struct {
	// ModelRoot is the DS host path of modelcatalog's root.
	ModelRoot string
	// ProvisionFile is the A-channel YAML table shipped with deployment
	// material; empty disables provisioning.
	ProvisionFile string
	// Prober runs OIP readiness checks; nil leaves records in VALIDATING.
	Prober ExternalProber
}

// ExternalRegistry exposes the merged provisioned/legacy-runtime health view
// and probes every record through the OIP prober. Managed replacement is
// disabled; existing persisted runtime rows remain visible after upgrade.
type ExternalRegistry struct {
	storePath string
	provision string
	prober    ExternalProber

	mu          sync.RWMutex
	provisioned map[string]ServableRecord
	runtime     map[string]ServableRecord
	states      map[string]externalProbeState

	stop      chan struct{}
	done      chan struct{}
	startOnce sync.Once
	stopOnce  sync.Once
}

type externalProbeState struct {
	state   string
	reason  string
	success int
	failure int
}

// NewExternal loads the provisioned and persisted tables without starting the
// probe loop. A missing provision or store file yields an empty registry.
func NewExternal(cfg ExternalConfig) (*ExternalRegistry, error) {
	if cfg.ModelRoot == "" {
		return nil, fmt.Errorf("external registry model root is required")
	}
	dir := filepath.Join(filepath.Clean(cfg.ModelRoot), filepath.FromSlash(externalDirRelPath))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	r := &ExternalRegistry{
		storePath:   filepath.Join(dir, externalStoreFile),
		provision:   cfg.ProvisionFile,
		prober:      cfg.Prober,
		provisioned: map[string]ServableRecord{},
		runtime:     map[string]ServableRecord{},
		states:      map[string]externalProbeState{},
		stop:        make(chan struct{}),
		done:        make(chan struct{}),
	}
	if err := r.loadProvisioned(); err != nil {
		return nil, err
	}
	if err := r.loadPersisted(); err != nil {
		return nil, err
	}
	return r, nil
}

// Start launches periodic probes of the static provisioned/legacy-runtime view.
func (r *ExternalRegistry) Start() {
	r.startOnce.Do(func() { go r.run() })
}

// Stop waits for the probe loop to terminate.
func (r *ExternalRegistry) Stop() {
	r.startOnce.Do(func() { close(r.done) })
	r.stopOnce.Do(func() { close(r.stop) })
	<-r.done
}

// Lookup returns a record from the merged table. Runtime records override
// provisioned records with the same servable id.
func (r *ExternalRegistry) Lookup(servableID string) (ServableRecord, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if rec, ok := r.runtime[servableID]; ok {
		return rec, true
	}
	rec, ok := r.provisioned[servableID]
	return rec, ok
}

// Snapshot returns the merged table in stable servable-id order with probe
// state and provisioning channel for each record.
func (r *ExternalRegistry) Snapshot() []ServableStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make(map[string]struct{}, len(r.provisioned)+len(r.runtime))
	for id := range r.provisioned {
		ids[id] = struct{}{}
	}
	for id := range r.runtime {
		ids[id] = struct{}{}
	}
	out := make([]ServableStatus, 0, len(ids))
	for id := range ids {
		channel := "provisioned"
		rec, ok := r.provisioned[id]
		if runtimeRec, isRuntime := r.runtime[id]; isRuntime {
			rec, channel = runtimeRec, "runtime"
		} else if !ok {
			continue
		}
		st := r.states[id]
		if st.state == "" {
			st.state = model.StateValidating
		}
		out = append(out, ServableStatus{
			ServableRecord: rec,
			Channel:        channel,
			State:          st.state,
			Reason:         st.reason,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ServableID < out[j].ServableID })
	return out
}

// Replace is a fail-closed compatibility guard for older callers. It performs
// no validation, persistence, map mutation or probe wakeup: every input,
// including nil and empty tables, has the same stable read-only error.
func (r *ExternalRegistry) Replace(_ []ServableRecord) error {
	return ErrExternalServablesReadOnly
}

// ValidateServableRecord enforces the external registry contract. Every
// cloud-derived string is whitelist-validated; nothing here is interpolated
// into shell commands or server configuration files.
func ValidateServableRecord(rec ServableRecord) error {
	if rec.Family != externalFamily {
		return fmt.Errorf("family must be %q, got %q", externalFamily, rec.Family)
	}
	if !model.IsSafeName(rec.ServableID) {
		return fmt.Errorf("servableId must be a safe name, got %q", rec.ServableID)
	}
	if rec.Protocol != externalProtocolKServeGRPC {
		return fmt.Errorf("protocol must be %q, got %q", externalProtocolKServeGRPC, rec.Protocol)
	}
	host, port, err := net.SplitHostPort(strings.TrimSpace(rec.Endpoint))
	if err != nil || host == "" || port == "" {
		return fmt.Errorf("endpoint must be host:port, got %q", rec.Endpoint)
	}
	if strings.Contains(rec.Endpoint, "://") {
		return fmt.Errorf("endpoint must be host:port without a scheme, got %q", rec.Endpoint)
	}
	if !model.IsSafeName(rec.ModelName) {
		return fmt.Errorf("modelName must be a safe name, got %q", rec.ModelName)
	}
	if rec.ModelVersion != "" && !model.IsSafeName(rec.ModelVersion) {
		return fmt.Errorf("modelVersion must be a safe name, got %q", rec.ModelVersion)
	}
	for _, device := range rec.Capabilities.Devices {
		if len(device) > 64 || !deviceRe.MatchString(device) {
			return fmt.Errorf("invalid capability device %q", device)
		}
	}
	if len(rec.Descriptor) > 0 {
		if err := validateExternalDescriptor(rec.Descriptor); err != nil {
			return err
		}
	}
	return nil
}

// validateExternalDescriptor checks only the sections needed to build
// requests: runtime backend/model name, tensor inputs and the decoder kind.
// Payload/checksum sections do not exist for remote servables.
func validateExternalDescriptor(raw json.RawMessage) error {
	var manifest model.Manifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return fmt.Errorf("parse descriptor: %w", err)
	}
	switch manifest.Runtime.Backend {
	case "kserve", "openai":
	default:
		return fmt.Errorf("descriptor runtime.backend must be kserve or openai, got %q", manifest.Runtime.Backend)
	}
	if manifest.Runtime.ModelName == "" || !model.IsSafeName(manifest.Runtime.ModelName) {
		return fmt.Errorf("descriptor runtime.model_name must be a safe name")
	}
	if manifest.Runtime.Backend == "kserve" {
		if len(manifest.IO.Inputs) == 0 {
			return fmt.Errorf("descriptor io.inputs must declare at least one tensor input")
		}
		if err := validateVisionInputSafe(manifest.IO.Inputs[0]); err != nil {
			return err
		}
	}
	if string(manifest.Decoder) == "" || string(manifest.Decoder) == "null" {
		return fmt.Errorf("descriptor decoder section is required")
	}
	var decoder struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(manifest.Decoder, &decoder); err != nil {
		return fmt.Errorf("parse decoder section: %w", err)
	}
	if decoder.Kind == "" {
		return fmt.Errorf("descriptor decoder.kind is required")
	}
	return nil
}

// validateVisionInputSafe re-checks the vision input contract without relying
// on the package-only manifest validation path.
func validateVisionInputSafe(in model.IOSpec) error {
	if in.Name == "" || in.DType != "FP32" || in.Layout != "NCHW" {
		return fmt.Errorf("vision input must be named FP32 NCHW, got name=%q dtype=%q layout=%q", in.Name, in.DType, in.Layout)
	}
	if len(in.Shape) != 4 || in.Shape[0] != 1 || in.Shape[1] != 3 || in.Shape[2] <= 0 || in.Shape[3] <= 0 {
		return fmt.Errorf("vision input shape must be [1,3,H,W], got %v", in.Shape)
	}
	return nil
}

func (r *ExternalRegistry) loadProvisioned() error {
	if r.provision == "" {
		return nil
	}
	b, err := os.ReadFile(r.provision)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read provisioned external servables: %w", err)
	}
	var table struct {
		Servables []ServableRecord `yaml:"servables"`
	}
	if err := yaml.Unmarshal(b, &table); err != nil {
		return fmt.Errorf("parse provisioned external servables %q: %w", r.provision, err)
	}
	provisioned := make(map[string]ServableRecord, len(table.Servables))
	for _, rec := range table.Servables {
		if err := ValidateServableRecord(rec); err != nil {
			return fmt.Errorf("provisioned external servable %q: %w", rec.ServableID, err)
		}
		if _, dup := provisioned[rec.ServableID]; dup {
			return fmt.Errorf("duplicate provisioned external servable id %q", rec.ServableID)
		}
		provisioned[rec.ServableID] = rec
	}
	r.provisioned = provisioned
	return nil
}

func (r *ExternalRegistry) loadPersisted() error {
	b, err := os.ReadFile(r.storePath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read persisted external servables: %w", err)
	}
	var records []ServableRecord
	if err := json.Unmarshal(b, &records); err != nil {
		return fmt.Errorf("parse persisted external servables: %w", err)
	}
	runtime := make(map[string]ServableRecord, len(records))
	for _, rec := range records {
		if err := ValidateServableRecord(rec); err != nil {
			return fmt.Errorf("persisted external servable %q: %w", rec.ServableID, err)
		}
		runtime[rec.ServableID] = rec
	}
	r.runtime = runtime
	return nil
}

func (r *ExternalRegistry) run() {
	defer close(r.done)
	r.scan(context.Background())
	ticker := time.NewTicker(externalProbeInterval)
	defer ticker.Stop()
	for {
		select {
		case <-r.stop:
			return
		case <-ticker.C:
			r.scan(context.Background())
		}
	}
}

func (r *ExternalRegistry) scan(parent context.Context) {
	r.mu.RLock()
	records := make([]ServableRecord, 0, len(r.provisioned)+len(r.runtime))
	seen := map[string]struct{}{}
	for id, rec := range r.runtime {
		records = append(records, rec)
		seen[id] = struct{}{}
	}
	for id, rec := range r.provisioned {
		if _, ok := seen[id]; ok {
			continue
		}
		records = append(records, rec)
	}
	r.mu.RUnlock()
	if len(records) == 0 {
		r.mu.Lock()
		r.states = map[string]externalProbeState{}
		r.mu.Unlock()
		return
	}

	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	sem := make(chan struct{}, externalProbeConcurrency)
	var wg sync.WaitGroup
	results := make(map[string]externalProbeState, len(records))
	var resultMu sync.Mutex
	for _, rec := range records {
		rec := rec
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			case <-r.stop:
				return
			}
			defer func() { <-sem }()
			resultMu.Lock()
			results[rec.ServableID] = r.probeOne(ctx, rec)
			resultMu.Unlock()
		}()
	}
	wg.Wait()

	r.mu.Lock()
	defer r.mu.Unlock()
	pruned := make(map[string]externalProbeState, len(results))
	for id, st := range results {
		pruned[id] = st
	}
	r.states = pruned
}

func (r *ExternalRegistry) probeOne(parent context.Context, rec ServableRecord) externalProbeState {
	r.mu.Lock()
	prev := r.states[rec.ServableID]
	r.mu.Unlock()
	if prev.state == "" {
		prev.state = model.StateValidating
	}
	if r.prober == nil {
		return prev
	}
	ctx, cancel := context.WithTimeout(parent, externalProbeTimeout)
	defer cancel()
	err := r.prober.Probe(ctx, rec.Protocol, rec.Endpoint, rec.ModelName, rec.ModelVersion)
	if err == nil {
		prev.success++
		prev.failure = 0
		prev.reason = ""
		if prev.state == model.StateReady || prev.success >= externalStableCount {
			prev.state = model.StateReady
		}
		return prev
	}
	prev.failure++
	prev.success = 0
	prev.reason = classifyExternalError(err)
	if prev.state == model.StateFailed || prev.failure >= externalStableCount {
		prev.state = model.StateFailed
	}
	return prev
}

// classifyExternalError maps probe failures onto the shared LastError
// vocabulary. Timeouts and connection failures are unreachable; a responding
// server that does not know the model yet is a not-ready condition.
func classifyExternalError(err error) string {
	if err == nil {
		return ""
	}
	text := err.Error()
	switch {
	case strings.Contains(text, "not ready"), strings.Contains(text, "not found"),
		strings.Contains(text, "NotFound"):
		return externalReasonModelNotReady
	case strings.Contains(text, "deadline"), strings.Contains(text, "timeout"),
		strings.Contains(text, "connection"), strings.Contains(text, "unavailable"),
		strings.Contains(text, "Unavailable"):
		return externalReasonRuntimeUnreach
	default:
		return externalReasonRuntimeError
	}
}
