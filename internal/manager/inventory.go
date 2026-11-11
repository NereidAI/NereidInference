// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package manager owns the service-level inference inventory exposed by the
// singleton EdgeX manager device. Job devices own one media/model pipeline
// each; this package aggregates the catalog, runtime readiness probes and
// binding counts into the full model inventory northbound integrations read.
package manager

import (
	"context"
	"errors"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/job"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
)

const (
	// probeInterval bounds steady-state runtime readiness refresh latency.
	probeInterval = 15 * time.Second
	// probeConcurrency bounds concurrent gRPC/HTTP probes independently from
	// the number of installed packages.
	probeConcurrency = 4
	// stableProbeCount implements the three-failure/three-success flap
	// guard required by the cross-product status contract.
	stableProbeCount = 3
)

// Model is one entry in the manager device ModelsJSON reading. Its JSON
// shape is the DS-local contract that NereidEdge translates to the ADR-0039
// manifest models segment; this service never imports NereidLink SDK types.
type Model struct {
	ID             string   `json:"id"`
	Version        string   `json:"version"`
	Revision       string   `json:"revision,omitempty"`
	Task           string   `json:"task"`
	Family         string   `json:"family,omitempty"`
	Backend        string   `json:"backend"`
	ModelName      string   `json:"model_name"`
	State          string   `json:"state"`
	Reason         string   `json:"reason,omitempty"`
	Labels         []string `json:"labels,omitempty"`
	ActiveBindings int      `json:"active_bindings,omitempty"`
}

// Config wires the service-level inventory to its shared owners.
type Config struct {
	Catalog        *modelcatalog.Catalog
	Backends       *infer.Registry
	KServeEndpoint string
	OpenAIEndpoint string
	// Servables optionally reports which managed families are enabled on
	// this node. Packages of a registered family are READY at catalog level;
	// unmanaged families keep going through runtime readiness probes.
	Servables *modelserver.Registry
}

// Inventory periodically verifies that every installed package is accepted
// by the configured runtime server. A package present on disk remains
// VALIDATING until the first stable runtime result; READY therefore retains
// the ADR meaning "runtime accepted the model", not merely "tarball valid".
type Inventory struct {
	catalog   *modelcatalog.Catalog
	backends  *infer.Registry
	servables *modelserver.Registry

	epMu           sync.RWMutex
	kserveEndpoint string
	openAIEndpoint string
	states         map[string]probeState
	probeMu        sync.Mutex
	stop           chan struct{}
	wakeCh         chan struct{}
	done           chan struct{}
	startOnce      sync.Once
	stopOnce       sync.Once
}

type probeState struct {
	state   string
	reason  string
	success int
	failure int
}

// New creates an inventory without starting its probe loop.
func New(cfg Config) *Inventory {
	return &Inventory{
		catalog:        cfg.Catalog,
		backends:       cfg.Backends,
		servables:      cfg.Servables,
		kserveEndpoint: cfg.KServeEndpoint,
		openAIEndpoint: cfg.OpenAIEndpoint,
		states:         map[string]probeState{},
		stop:           make(chan struct{}),
		wakeCh:         make(chan struct{}, 1),
		done:           make(chan struct{}),
	}
}

// Start launches the initial probe and periodic/catalog-event refreshes.
func (s *Inventory) Start() {
	s.startOnce.Do(func() {
		events, unsubscribe := s.catalog.Subscribe()
		go s.run(events, unsubscribe)
	})
}

// Stop waits for the probe loop to terminate. Callers must close backends
// only after Stop returns.
func (s *Inventory) Stop() {
	s.Start()
	s.stopOnce.Do(func() { close(s.stop) })
	<-s.done
}

// SetEndpoints updates runtime endpoints selected by service configuration.
// The next probe uses the new values without waiting for a restart.
func (s *Inventory) SetEndpoints(kserveEndpoint, openAIEndpoint string) {
	s.epMu.Lock()
	s.kserveEndpoint = kserveEndpoint
	s.openAIEndpoint = openAIEndpoint
	s.epMu.Unlock()
	s.wake()
}

// BindingKey returns the stable map key used for one model binding count.
func BindingKey(id, version string) string { return id + "\x00" + version }

// Models returns the full installed-model inventory in catalog order.
// bindings counts currently running job devices by model id and resolved
// version; a floating model version is counted against its installed entry.
// ObjectDiagnostics returns source objects ignored or rejected by the catalog.
func (s *Inventory) ObjectDiagnostics() modelcatalog.ObjectDiagnostics {
	return s.catalog.ObjectDiagnostics()
}

func (s *Inventory) Models(bindings map[string]int) []Model {
	entries := s.catalog.SortedEntries()
	out := make([]Model, 0, len(entries))
	for _, entry := range entries {
		key := BindingKey(entry.ID, entry.Version)
		state := s.stateFor(key)
		// Managed-family runtime readiness is scoped to a binding's target
		// device and reconciled by the registered servable manager. The
		// immutable package is READY as soon as catalog validation succeeds;
		// the Job device reports CPU/GPU servable readiness. Packages whose
		// family is not managed on this node keep being probed directly.
		if entry.Manifest != nil && s.isManaged(entry.Manifest.Runtime.Server.Family) {
			state = probeState{state: model.StateReady}
		}
		labels := entry.Labels
		if len(labels) == 0 {
			labels = nil
		}
		out = append(out, Model{
			ID:             entry.ID,
			Version:        entry.Version,
			Revision:       entry.Revision,
			Task:           entry.Manifest.Model.Task,
			Family:         entry.Manifest.Model.Family,
			Backend:        entry.Manifest.Runtime.Backend,
			ModelName:      entry.Manifest.Runtime.ModelName,
			State:          state.state,
			Reason:         state.reason,
			Labels:         labels,
			ActiveBindings: bindings[key],
		})
	}
	return out
}

// isManaged reports whether a family token has a registered servable manager
// on this node; a nil registry means every family is probed directly.
func (s *Inventory) isManaged(family string) bool {
	if s.servables == nil || !modelserver.IsManagedFamily(family) {
		return false
	}
	_, ok := s.servables.For(family)
	return ok
}

func (s *Inventory) stateFor(key string) probeState {
	s.probeMu.Lock()
	defer s.probeMu.Unlock()
	if st, ok := s.states[key]; ok {
		return st
	}
	return probeState{state: model.StateValidating}
}

func (s *Inventory) run(events <-chan modelcatalog.Event, unsubscribe func()) {
	defer close(s.done)
	defer unsubscribe()

	scan := make(chan struct{}, 1)
	trigger := func() {
		select {
		case scan <- struct{}{}:
		default:
		}
	}
	trigger()

	ticker := time.NewTicker(probeInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-scan:
			s.scan(context.Background())
		case <-events:
			// Coalesce catalog bursts; the scan reads the authoritative
			// catalog rather than trusting individual event payloads.
			trigger()
		case <-s.wakeCh:
			trigger()
		case <-ticker.C:
			trigger()
		}
	}
}

func (s *Inventory) wake() {
	select {
	case s.wakeCh <- struct{}{}:
	default:
	}
}

func (s *Inventory) scan(parent context.Context) {
	entries := s.catalog.SortedEntries()
	if len(entries) == 0 {
		s.prune(nil)
		return
	}

	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	sem := make(chan struct{}, probeConcurrency)
	var wg sync.WaitGroup
	results := make(map[string]probeState, len(entries))
	var resultMu sync.Mutex

	for _, entry := range entries {
		entry := entry
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return
			case <-s.stop:
				return
			}
			defer func() { <-sem }()
			result := s.probeEntry(ctx, entry)
			resultMu.Lock()
			results[BindingKey(entry.ID, entry.Version)] = result
			resultMu.Unlock()
		}()
	}
	wg.Wait()
	s.prune(results)
}

func (s *Inventory) probeEntry(ctx context.Context, entry modelcatalog.Entry) probeState {
	if entry.Manifest != nil && s.isManaged(entry.Manifest.Runtime.Server.Family) {
		return s.recordResult(BindingKey(entry.ID, entry.Version), true, "")
	}
	endpoint, err := s.endpointFor(entry.Manifest)
	if err != nil {
		return s.recordResult(BindingKey(entry.ID, entry.Version), false, job.ReasonConfigInvalid)
	}
	backend, err := s.backends.Get(entry.Manifest.Runtime.Backend, endpoint)
	if err != nil {
		return s.recordResult(BindingKey(entry.ID, entry.Version), false, job.ReasonConfigInvalid)
	}
	probeCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	err = backend.Ready(probeCtx, entry.Manifest.Runtime.ModelName, "")
	if err == nil {
		return s.recordResult(BindingKey(entry.ID, entry.Version), true, "")
	}
	return s.recordResult(BindingKey(entry.ID, entry.Version), false, classifyReadyError(err))
}

func (s *Inventory) endpointFor(m *model.Manifest) (string, error) {
	s.epMu.RLock()
	defer s.epMu.RUnlock()
	switch m.Runtime.Backend {
	case "kserve":
		if s.kserveEndpoint == "" {
			return "", errors.New("kserve endpoint is empty")
		}
		return s.kserveEndpoint, nil
	case "openai":
		if s.openAIEndpoint == "" {
			return "", errors.New("openai endpoint is empty")
		}
		return s.openAIEndpoint, nil
	default:
		return "", errors.New("unsupported backend " + m.Runtime.Backend)
	}
}

func (s *Inventory) recordResult(key string, ready bool, reason string) probeState {
	s.probeMu.Lock()
	defer s.probeMu.Unlock()
	prev := s.states[key]
	if prev.state == "" {
		prev.state = model.StateValidating
	}
	if ready {
		prev.success++
		prev.failure = 0
		prev.reason = ""
		if prev.state == model.StateReady || prev.success >= stableProbeCount {
			prev.state = model.StateReady
		}
		s.states[key] = prev
		return prev
	}
	prev.failure++
	prev.success = 0
	prev.reason = reason
	if prev.state == model.StateFailed || prev.failure >= stableProbeCount {
		prev.state = model.StateFailed
	}
	s.states[key] = prev
	return prev
}

func (s *Inventory) prune(next map[string]probeState) {
	s.probeMu.Lock()
	defer s.probeMu.Unlock()
	if next == nil {
		s.states = map[string]probeState{}
		return
	}
	pruned := make(map[string]probeState, len(next))
	for key, state := range next {
		pruned[key] = state
	}
	s.states = pruned
}

func classifyReadyError(err error) string {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return job.ReasonRuntimeUnreach
	}
	switch status.Code(err) {
	case codes.Unavailable, codes.DeadlineExceeded, codes.Canceled, codes.ResourceExhausted:
		return job.ReasonRuntimeUnreach
	case codes.NotFound:
		return job.ReasonModelNotReady
	default:
		return job.ReasonRuntimeError
	}
}
