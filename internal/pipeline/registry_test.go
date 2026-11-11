// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestRegistryStartsEmptyWithoutProvision(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if got := registry.Snapshot(); len(got) != 0 {
		t.Fatalf("registry without provisioning must start empty, got %#v", got)
	}
}

func TestSampleProvisionFileLoadsFaceRecognitionDefinition(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{
		ModelRoot: t.TempDir(), ProvisionFile: filepath.Join("..", "..", "samples", "pipelines", "face-recognition-v1.yaml"),
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, ok := registry.Lookup(faceRecognitionID)
	if !ok {
		t.Fatal("sample face pipeline is not registered from the provision file")
	}
	if !definitionsEqual(definition, faceRecognitionDefinition()) {
		t.Fatalf("provisioned sample differs from test fixture:\n%#v\n%#v", definition, faceRecognitionDefinition())
	}
	if got := len(definition.Nodes); got != 5 {
		t.Fatalf("nodes = %d, want 5", got)
	}
}

func TestRegistryAtomicReplaceImmutabilityAndPersistence(t *testing.T) {
	root := t.TempDir()
	registry, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	builtIn := faceRecognitionDefinition()
	custom := faceRecognitionDefinition()
	custom.PipelineID = "custom-face"
	if err := registry.Replace([]Definition{builtIn, custom}); err != nil {
		t.Fatal(err)
	}
	before := registry.Snapshot()
	store := filepath.Join(root, filepath.FromSlash(pipelineStoreRelPath))
	persistedBefore, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}

	mutated := custom
	mutated.Nodes = append([]Node(nil), custom.Nodes...)
	mutated.Nodes[2].Mode = "roi"
	if err := registry.Replace([]Definition{builtIn, mutated}); err == nil {
		t.Fatal("expected immutable id+version modification to be rejected")
	}
	if got := registry.Snapshot(); !reflect.DeepEqual(got, before) {
		t.Fatalf("immutable rejection changed live table:\n%#v\n%#v", got, before)
	}

	invalid := custom
	invalid.PipelineID = "invalid-table-entry"
	invalid.Edges = [][2]string{{"source", "emit"}}
	if err := registry.Replace([]Definition{builtIn, invalid}); err == nil {
		t.Fatal("expected invalid definition to reject the whole table")
	}
	if got := registry.Snapshot(); !reflect.DeepEqual(got, before) {
		t.Fatal("invalid table partially changed the live registry")
	}
	persistedAfter, err := os.ReadFile(store)
	if err != nil {
		t.Fatal(err)
	}
	if string(persistedAfter) != string(persistedBefore) {
		t.Fatal("rejected table changed the persistent file")
	}

	reloaded, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Snapshot(); !reflect.DeepEqual(got, before) {
		t.Fatalf("persisted table did not reload:\n%#v\n%#v", got, before)
	}
}

func TestRegistryDeletedVersionRemainsImmutableAcrossReplacementAndRestart(t *testing.T) {
	root := t.TempDir()
	registry, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	original := faceRecognitionDefinition()
	original.PipelineID = "  historical  "
	original.Nodes[1].Model.ModelID = " yunet "
	original.Nodes[1].Model.TargetDevice = " cpu "
	original.Nodes[3].Model.TargetDevice = "   "
	original.Nodes[2].Mode = ""
	original.Limits = Limits{}
	if err := registry.Replace([]Definition{original}); err != nil {
		t.Fatal(err)
	}
	if err := registry.Replace([]Definition{}); err != nil {
		t.Fatal(err)
	}
	reloaded, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	mutated := cloneDefinition(original)
	mutated.Description = "remove-and-re-add must not redefine this version"
	if err := reloaded.Replace([]Definition{mutated}); err == nil {
		t.Fatal("deleted version was re-added with different content after restart")
	}
	equivalent := faceRecognitionDefinition()
	equivalent.PipelineID = "historical"
	equivalent.Nodes[1].Model.TargetDevice = "CPU"
	if err := reloaded.Replace([]Definition{equivalent}); err != nil {
		t.Fatalf("canonically equivalent historical definition should be accepted: %v", err)
	}
	if err := reloaded.Replace([]Definition{original}); err != nil {
		t.Fatalf("original noncanonical historical definition should replay canonically: %v", err)
	}
}

func TestRegistryVersionAliasesHaveDeterministicOrder(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	first := faceRecognitionDefinition()
	first.PipelineID = "aliases"
	first.Version = "1.01.0"
	second := cloneDefinition(first)
	second.Version = "1.1.0"
	if err := registry.Replace([]Definition{first, second}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		got, ok := registry.Lookup("aliases")
		if !ok || got.Version != "1.1.0" {
			t.Fatalf("lookup %d = %q, %v; want deterministic lexical tie-break 1.1.0", i, got.Version, ok)
		}
	}
	snapshot := registry.Snapshot()
	if len(snapshot) != 2 || snapshot[0].Version != "1.01.0" || snapshot[1].Version != "1.1.0" {
		t.Fatalf("unstable snapshot order: %#v", snapshot)
	}
}

func TestRegistryLatestVersionAndWholeTableReplacement(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	v1 := faceRecognitionDefinition()
	v1.PipelineID = "versioned"
	v1.Version = "1.9.0"
	v2 := v1
	v2.Version = "1.10.0"
	if err := registry.Replace([]Definition{v1, v2}); err != nil {
		t.Fatal(err)
	}
	got, ok := registry.Lookup("versioned")
	if !ok || got.Version != "1.10.0" {
		t.Fatalf("latest = %q, %v; want 1.10.0", got.Version, ok)
	}
	updates, unsubscribe := registry.Subscribe()
	defer unsubscribe()
	if err := registry.Replace([]Definition{v1}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-updates:
	default:
		t.Fatal("replacement did not notify subscribers")
	}
	if got := registry.Snapshot(); len(got) != 1 || got[0].Version != "1.9.0" {
		t.Fatalf("full replacement did not remove old versions: %#v", got)
	}
	if got, ok := registry.Lookup("versioned"); !ok || got.Version != "1.9.0" {
		t.Fatalf("rollback lookup = %q, %v; want newest remaining 1.9.0", got.Version, ok)
	}
	// Re-adding an older version while a newer one remains does not itself roll
	// execution back: PipelineId-only lookup deliberately selects latest.
	if err := registry.Replace([]Definition{v1, v2}); err != nil {
		t.Fatal(err)
	}
	if got, _ := registry.Lookup("versioned"); got.Version != "1.10.0" {
		t.Fatalf("reaccepted older version overrode newer latest: %q", got.Version)
	}
}

func TestVersionComparisonIsTransitiveAndArbitraryPrecision(t *testing.T) {
	versions := []string{
		"1", "1.0", "1.0.0", "1.0.0-alpha", "1.01.0", "1.1.0",
		"2", "9", "10", "1a", "1..0", "1.0.a", "1+build", "1-1",
		"18446744073709551615", "18446744073709551616", "A", "a",
	}
	sign := func(value int) int {
		if value < 0 {
			return -1
		}
		if value > 0 {
			return 1
		}
		return 0
	}
	for _, a := range versions {
		for _, b := range versions {
			ab, ba := sign(compareVersionIdentity(a, b)), sign(compareVersionIdentity(b, a))
			if ab != -ba {
				t.Fatalf("antisymmetry failed for %q/%q: %d/%d", a, b, ab, ba)
			}
			if (a == b) != (ab == 0) {
				t.Fatalf("identity tie-break failed for %q/%q: %d", a, b, ab)
			}
			for _, c := range versions {
				if ab < 0 && compareVersionIdentity(b, c) < 0 && compareVersionIdentity(a, c) >= 0 {
					t.Fatalf("transitivity failed: %q < %q < %q", a, b, c)
				}
			}
		}
	}
	for _, test := range []struct {
		a, b string
		want int
	}{
		{"1.9.0", "1.10.0", -1},
		{"1.0.0", "1.0.0-alpha", -1},
		{"1.01.0", "1.1.0", -1},
		{"1", "1.0", -1},
		{"9", "18446744073709551616", -1},
		{"18446744073709551615", "18446744073709551616", -1},
		{"2", "10", -1},
		{"10", "1a", -1},
	} {
		if got := sign(compareVersionIdentity(test.a, test.b)); got != test.want {
			t.Errorf("compareVersionIdentity(%q, %q) = %d, want %d", test.a, test.b, got, test.want)
		}
	}
}

func TestRegistryLatestSuffixAndTotalOrderAcrossInsertions(t *testing.T) {
	versions := []string{"2", "10", "1.0.0", "1.0.0-alpha", "18446744073709551616"}
	for rotation := range versions {
		root := t.TempDir()
		registry, err := NewRegistry(RegistryConfig{ModelRoot: root})
		if err != nil {
			t.Fatal(err)
		}
		records := make([]Definition, 0, len(versions))
		for i := range versions {
			definition := faceRecognitionDefinition()
			definition.PipelineID = "ordering"
			definition.Version = versions[(i+rotation)%len(versions)]
			// No packages are installed: acceptance and latest selection are
			// structural and intentionally independent of runtime READY.
			records = append(records, definition)
		}
		if err := registry.Replace(records); err != nil {
			t.Fatal(err)
		}
		got, ok := registry.Lookup("ordering")
		if !ok || got.Version != "18446744073709551616" {
			t.Fatalf("rotation %d latest = %q, %v; want arbitrary-precision numeric latest", rotation, got.Version, ok)
		}
		first := registry.Snapshot()
		reloaded, err := NewRegistry(RegistryConfig{ModelRoot: root})
		if err != nil {
			t.Fatal(err)
		}
		if second := reloaded.Snapshot(); !reflect.DeepEqual(second, first) {
			t.Fatalf("rotation %d changed order after restart:\n%+v\n%+v", rotation, first, second)
		}
		if latest, _ := reloaded.Lookup("ordering"); latest.Version != got.Version {
			t.Fatalf("rotation %d restart latest = %q, want %q", rotation, latest.Version, got.Version)
		}
	}
}

func TestRegistrySuffixRollbackSelectsNewestRemaining(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	base := faceRecognitionDefinition()
	base.PipelineID = "suffix"
	suffix := cloneDefinition(base)
	suffix.Version = "1.0.0-alpha"
	if err := registry.Replace([]Definition{base, suffix}); err != nil {
		t.Fatal(err)
	}
	if got, _ := registry.Lookup("suffix"); got.Version != suffix.Version {
		t.Fatalf("suffix latest = %q, want %q", got.Version, suffix.Version)
	}
	if err := registry.Replace([]Definition{base}); err != nil {
		t.Fatal(err)
	}
	if got, _ := registry.Lookup("suffix"); got.Version != base.Version {
		t.Fatalf("suffix rollback = %q, want %q", got.Version, base.Version)
	}
}

func TestRegistryConcurrentCompetingReplacementKeepsMemoryDiskAndHistoryConsistent(t *testing.T) {
	root := t.TempDir()
	registry, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	left := faceRecognitionDefinition()
	left.PipelineID = "shared"
	left.Description = "left complete table"
	right := cloneDefinition(left)
	right.Description = "right complete table"

	start := make(chan struct{})
	errs := make(chan error, 2)
	var wg sync.WaitGroup
	for _, definition := range []Definition{left, right} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs <- registry.Replace([]Definition{definition})
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	var successes, failures int
	for err := range errs {
		if err == nil {
			successes++
		} else {
			failures++
		}
	}
	if successes != 1 || failures != 1 {
		t.Fatalf("competing immutable replacements = %d successes/%d failures, want 1/1", successes, failures)
	}
	visible := registry.Snapshot()
	if len(visible) != 1 || (visible[0].Description != left.Description && visible[0].Description != right.Description) {
		t.Fatalf("final table is not exactly one complete contender: %+v", visible)
	}
	reloaded, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Snapshot(); !reflect.DeepEqual(got, visible) {
		t.Fatalf("disk/memory diverged after concurrent replacement:\n%+v\n%+v", got, visible)
	}
	loser := right
	if visible[0].Description == right.Description {
		loser = left
	}
	if err := reloaded.Replace([]Definition{loser}); err == nil {
		t.Fatal("losing concurrent definition bypassed immutable history after restart")
	}
}

func TestRegistryConcurrentValidReplacementsRemainWholeTables(t *testing.T) {
	root := t.TempDir()
	registry, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	left := faceRecognitionDefinition()
	left.PipelineID = "left-only"
	right := faceRecognitionDefinition()
	right.PipelineID = "right-only"
	start := make(chan struct{})
	errs := make(chan error, 2)
	var wg sync.WaitGroup
	for _, definition := range []Definition{left, right} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs <- registry.Replace([]Definition{definition})
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("disjoint valid replacement failed: %v", err)
		}
	}
	visible := registry.Snapshot()
	if len(visible) != 1 || (visible[0].PipelineID != left.PipelineID && visible[0].PipelineID != right.PipelineID) {
		t.Fatalf("concurrent replacements merged or partially applied tables: %+v", visible)
	}
	reloaded, err := NewRegistry(RegistryConfig{ModelRoot: root})
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Snapshot(); !reflect.DeepEqual(got, visible) {
		t.Fatalf("valid concurrent replacements left disk/memory drift:\n%+v\n%+v", got, visible)
	}
}

func TestRegistryRejectsMalformedPersistedTables(t *testing.T) {
	for name, contents := range map[string]string{
		"null":              "null",
		"object":            "{}",
		"missing pipelines": `{"history":{}}`,
		"null pipelines":    `{"pipelines":null,"history":{}}`,
		"missing history":   `{"pipelines":[]}`,
		"null history":      `{"pipelines":[],"history":null}`,
		"unknown field":     `{"pipelines":[],"history":{},"unknown":true}`,
		"unknown nested":    `{"pipelines":[{"pipelineId":"x","unknown":true}],"history":{}}`,
		"trailing":          `{"pipelines":[],"history":{}} []`,
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			store := filepath.Join(root, filepath.FromSlash(pipelineStoreRelPath))
			if err := os.MkdirAll(filepath.Dir(store), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(store, []byte(contents), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := NewRegistry(RegistryConfig{ModelRoot: root}); err == nil {
				t.Fatalf("invalid persisted state was accepted as a registry: %s", contents)
			}
		})
	}
}

func TestRegistryRejectsInvalidAndDuplicateDefinitions(t *testing.T) {
	registry, err := NewRegistry(RegistryConfig{ModelRoot: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	valid := faceRecognitionDefinition()
	for name, mutate := range map[string]func(*Definition){
		"unknown node":        func(d *Definition) { d.Nodes[1].Kind = "script" },
		"missing output refs": func(d *Definition) { d.Nodes[1].Model.ModelID = "" },
		"bad transform":       func(d *Definition) { d.Nodes[2].Operator = "python" },
		"nonlinear":           func(d *Definition) { d.Edges[1] = [2]string{"source", "emit"} },
		"zero objects":        func(d *Definition) { d.Limits.MaxObjects = -1 },
		"duplicate":           func(d *Definition) {},
	} {
		t.Run(name, func(t *testing.T) {
			candidate := cloneDefinition(valid)
			mutate(&candidate)
			records := []Definition{candidate}
			if name == "duplicate" {
				records = append(records, cloneDefinition(candidate))
			}
			if err := registry.Replace(records); err == nil {
				t.Fatal("expected registry validation error")
			}
		})
	}
}
