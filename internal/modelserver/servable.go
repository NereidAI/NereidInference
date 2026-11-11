// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

// Descriptor runtime.server.family tokens. Vendor names never appear outside
// family adapters and their descriptors; generic code dispatches on these.
const (
	// FamilyOVMS is the node-local, NI-managed OpenVINO Model Server.
	FamilyOVMS = "ovms"
	// FamilyTriton is the node-local, NI-managed Triton inference server.
	FamilyTriton = "triton"
	// FamilyExternal marks servers NI cannot administrate; records live in
	// the external registry rather than behind a ServableManager.
	FamilyExternal = "external"
)

// ModelRef identifies one distributed model version held by a lease.
type ModelRef struct {
	ID      string
	Version string
}

// Lease is one binding's reference to a ready model-server servable. Lease
// implementations are concurrency-safe; Release is idempotent.
type Lease interface {
	// ModelName is the model name to put in dimension-A backend requests.
	ModelName() string
	// Endpoint returns the backend endpoint (host:port or base URL) of the
	// server hosting the servable. Managed leases must return a non-empty
	// endpoint; an explicit job override may replace it, but a service default
	// must never substitute for a missing lease endpoint.
	Endpoint() string
	// Ref identifies the distributed package the servable was rendered from.
	Ref() ModelRef
	// Current reports whether the lease still matches the manager's active
	// generation. Package removal or a re-render that drops the servable
	// invalidates the lease; the holder must Release and re-Acquire instead
	// of polling a name the server no longer serves.
	Current() bool
	// Release drops one binding reference. It is safe to call more than once.
	Release()
}

// ServableManager turns one distributed model package into a ready,
// lease-able endpoint on one node-local model server. Implementations are
// selected purely by the descriptor family token; repeated Acquire calls for
// the same (model, version, target device) identity share one servable.
type ServableManager interface {
	// Family is the descriptor token this manager owns.
	Family() string
	// Acquire ensures a servable for the package and target device exists and
	// returns one lease for the calling binding.
	Acquire(ctx context.Context, entry *modelcatalog.Entry, targetDevice string) (Lease, error)
	// RemoveModel unloads every target-device servable of one model version
	// before the catalog deletes its files. A failed server call must not
	// block package revocation.
	RemoveModel(id, version string)
	// Start begins catalog-event driven reconciliation.
	Start()
	// Stop terminates reconciliation. Active bindings are not removed.
	Stop()
}

// IsManagedFamily reports whether a descriptor family is reconciled by a
// ServableManager on this node. External and unknown families are not.
func IsManagedFamily(family string) bool {
	switch family {
	case FamilyOVMS, FamilyTriton:
		return true
	default:
		return false
	}
}

// Registry dispatches servable operations to the manager registered for the
// descriptor family token, mirroring the infer.Backend registry pattern.
type Registry struct {
	mu       sync.RWMutex
	managers map[string]ServableManager
}

// NewRegistry builds a family registry. Managers may be passed up front or
// added later through Register.
func NewRegistry(managers ...ServableManager) *Registry {
	r := &Registry{managers: map[string]ServableManager{}}
	for _, m := range managers {
		if err := r.Register(m); err != nil {
			panic(err)
		}
	}
	return r
}

// Register adds one family manager. Duplicate families are rejected.
func (r *Registry) Register(m ServableManager) error {
	if m == nil {
		return fmt.Errorf("cannot register a nil servable manager")
	}
	family := m.Family()
	if family == "" {
		return fmt.Errorf("servable manager family must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.managers[family]; exists {
		return fmt.Errorf("servable manager for family %q is already registered", family)
	}
	r.managers[family] = m
	return nil
}

// For returns the manager registered for a descriptor family token.
func (r *Registry) For(family string) (ServableManager, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.managers[family]
	return m, ok
}

// Families lists the registered family tokens in stable order.
func (r *Registry) Families() []string {
	r.mu.RLock()
	out := make([]string, 0, len(r.managers))
	for family := range r.managers {
		out = append(out, family)
	}
	r.mu.RUnlock()
	sort.Strings(out)
	return out
}

// Acquire resolves the package's family and delegates to its manager. A
// package whose family has no registered manager is a bind-time configuration
// error rather than a silent fallback.
func (r *Registry) Acquire(ctx context.Context, entry *modelcatalog.Entry, targetDevice string) (Lease, error) {
	if entry == nil || entry.Manifest == nil {
		return nil, fmt.Errorf("model entry is required")
	}
	family := entry.Manifest.Runtime.Server.Family
	m, ok := r.For(family)
	if !ok {
		return nil, fmt.Errorf("no servable manager registered for family %q", family)
	}
	return m.Acquire(ctx, entry, targetDevice)
}

// RemoveModel fans the catalog pre-removal hook out to every manager so each
// server unloads the model before its on-disk files are deleted.
func (r *Registry) RemoveModel(id, version string) {
	r.mu.RLock()
	managers := make([]ServableManager, 0, len(r.managers))
	for _, m := range r.managers {
		managers = append(managers, m)
	}
	r.mu.RUnlock()
	for _, m := range managers {
		m.RemoveModel(id, version)
	}
}

// Start starts every registered manager.
func (r *Registry) Start() {
	r.mu.RLock()
	managers := make([]ServableManager, 0, len(r.managers))
	for _, m := range r.managers {
		managers = append(managers, m)
	}
	r.mu.RUnlock()
	for _, m := range managers {
		m.Start()
	}
}

// Stop stops every registered manager.
func (r *Registry) Stop() {
	r.mu.RLock()
	managers := make([]ServableManager, 0, len(r.managers))
	for _, m := range r.managers {
		managers = append(managers, m)
	}
	r.mu.RUnlock()
	for _, m := range managers {
		m.Stop()
	}
}
