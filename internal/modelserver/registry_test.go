// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"context"
	"sync"
	"testing"

	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

func manifestForFamily(family string) *model.Manifest {
	m := &model.Manifest{}
	m.Runtime.Server.Family = family
	return m
}

type fakeManager struct {
	mu         sync.Mutex
	family     string
	acquires   int
	releases   int
	removed    []string
	started    bool
	stopped    bool
	acquireErr error
}

func (f *fakeManager) Family() string { return f.family }

func (f *fakeManager) Acquire(_ context.Context, entry *modelcatalog.Entry, _ string) (Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.acquires++
	if f.acquireErr != nil {
		return nil, f.acquireErr
	}
	return &fakeLease{manager: f, ref: ModelRef{ID: entry.ID, Version: entry.Version}}, nil
}

func (f *fakeManager) RemoveModel(id, version string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.removed = append(f.removed, id+"/"+version)
}

func (f *fakeManager) Start() {
	f.mu.Lock()
	f.started = true
	f.mu.Unlock()
}

func (f *fakeManager) Stop() {
	f.mu.Lock()
	f.stopped = true
	f.mu.Unlock()
}

type fakeLease struct {
	manager *fakeManager
	ref     ModelRef
	once    sync.Once
}

func (l *fakeLease) ModelName() string { return "fake-model" }
func (l *fakeLease) Endpoint() string  { return "fake:9001" }
func (l *fakeLease) Ref() ModelRef     { return l.ref }
func (l *fakeLease) Current() bool     { return true }
func (l *fakeLease) Release() {
	l.once.Do(func() {
		l.manager.mu.Lock()
		l.manager.releases++
		l.manager.mu.Unlock()
	})
}

func TestRegistryDispatchesByFamily(t *testing.T) {
	ovms := &fakeManager{family: FamilyOVMS}
	triton := &fakeManager{family: FamilyTriton}
	registry := NewRegistry(ovms, triton)

	if got := registry.Families(); len(got) != 2 || got[0] != FamilyOVMS || got[1] != FamilyTriton {
		t.Fatalf("families = %v, want [ovms triton]", got)
	}
	if err := registry.Register(&fakeManager{family: FamilyOVMS}); err == nil {
		t.Fatal("duplicate family registration must fail")
	}
	if _, ok := registry.For("unknown-family"); ok {
		t.Fatal("unknown family must not resolve")
	}
	entry := &modelcatalog.Entry{ID: "m1", Version: "1.0.0", Manifest: manifestForFamily(FamilyTriton)}
	lease, err := registry.Acquire(context.Background(), entry, "CPU")
	if err != nil {
		t.Fatal(err)
	}
	if triton.acquires != 1 || ovms.acquires != 0 {
		t.Fatalf("acquire dispatch wrong: triton=%d ovms=%d", triton.acquires, ovms.acquires)
	}
	if lease.Endpoint() != "fake:9001" || lease.Ref().ID != "m1" {
		t.Fatalf("lease mismatch: %+v", lease)
	}

	registry.RemoveModel("m1", "1.0.0")
	if len(triton.removed) != 1 || len(ovms.removed) != 1 {
		t.Fatalf("remove must fan out to every manager: triton=%v ovms=%v", triton.removed, ovms.removed)
	}
	registry.Start()
	registry.Stop()
	if !ovms.started || !triton.started || !ovms.stopped || !triton.stopped {
		t.Fatal("start/stop must fan out to every manager")
	}
}

func TestRegistryAcquireRejectsUnmanagedFamily(t *testing.T) {
	registry := NewRegistry()
	entry := &modelcatalog.Entry{ID: "m1", Version: "1.0.0", Manifest: manifestForFamily("mystery")}
	if _, err := registry.Acquire(context.Background(), entry, "CPU"); err == nil {
		t.Fatal("unmanaged family must fail at bind time, not silently fall back")
	}
}

func TestIsManagedFamily(t *testing.T) {
	for _, family := range []string{FamilyOVMS, FamilyTriton} {
		if !IsManagedFamily(family) {
			t.Fatalf("%q must be managed", family)
		}
	}
	for _, family := range []string{FamilyExternal, "", "nvidia"} {
		if IsManagedFamily(family) {
			t.Fatalf("%q must not be managed", family)
		}
	}
}
