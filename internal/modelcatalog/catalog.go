// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package modelcatalog is the single writer for model packages on the node.
// It watches the Open Horizon ESS local-object directory and/or a plain inbox
// drop directory, accepts the standardized gzip tar Nereid model package,
// validates its descriptor and payload checksums, and atomically installs
// unpacked packages into an active model root shared with runtime adapters.
// Catalog validation is runtime-neutral; an adapter rejects artifacts it
// cannot serve. Removal of a source object removes the active model and
// notifies subscribers.
//
// The catalog never talks to a server or a cloud; it owns files only. Jobs
// consume the read-only active tree and perform ModelReady gating themselves.
package modelcatalog

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"

	"github.com/edgexfoundry/device-inference/internal/model"
)

// Config configures the watch roots.
type Config struct {
	Root         string // catalog root (staging/active/ledger.json)
	EssEnabled   bool
	EssDir       string
	InboxEnabled bool
	InboxDir     string
	// BeforeRemove is invoked after catalog state is detached but before the
	// active files are deleted, allowing the runtime server to unload.
	BeforeRemove func(id, version string)
	// Log receives one object-accept/ignore/reject line. Nil disables logging;
	// object diagnostics remain available through ObjectDiagnostics.
	Log Logger
}

// Logger is the minimal catalog logging surface.
type Logger interface {
	Debugf(format string, args ...any)
	Infof(format string, args ...any)
	Errorf(format string, args ...any)
}

// ObjectDiagnostic explains one source object that is not currently installed.
type ObjectDiagnostic struct {
	Name      string    `json:"name"`
	State     string    `json:"state"`
	Reason    string    `json:"reason"`
	SizeBytes int64     `json:"size_bytes,omitempty"`
	Attempts  int64     `json:"attempts"`
	LastSeen  time.Time `json:"last_seen"`
}

// ObjectDiagnostics is the northbound summary of ignored/rejected ESS objects.
type ObjectDiagnostics struct {
	IgnoredCount  int                `json:"ignored_count"`
	RejectedCount int                `json:"rejected_count"`
	Objects       []ObjectDiagnostic `json:"objects"`
}

const maxObjectDiagnostics = 20

// Entry is one installed model version.
type Entry struct {
	ID       string
	Version  string
	Revision string
	Dir      string
	Manifest *model.Manifest
	Labels   []string
}

// EventKind distinguishes catalog updates.
type EventKind string

// Event kinds.
const (
	EventAdded   EventKind = "added"
	EventRemoved EventKind = "removed"
)

// Event notifies jobs about active-tree changes.
type Event struct {
	Kind    EventKind
	ModelID string
	Version string
}

type eventSubscription struct {
	ch   chan Event
	done chan struct{}
}

// Catalog owns the active model tree.
type Catalog struct {
	cfg Config

	mu      sync.RWMutex
	entries map[string]*Entry // key id + "\x00" + version
	ledger  ledger

	watcher *fsnotify.Watcher
	stop    chan struct{}
	done    chan struct{}

	startMu  sync.Mutex
	running  bool
	stopOnce sync.Once

	subMu sync.Mutex
	subs  map[*eventSubscription]struct{}

	// tried records one-shot tar-probe outcomes so unrelated ESS objects
	// (e.g. non-model MMS payloads) are not reopened forever.
	tried map[string]probeOutcome

	// diagnostics explains the latest ignored/rejected source objects.
	diagMu sync.Mutex
	diag   map[string]*ObjectDiagnostic
}

type probeOutcome struct {
	ok   bool
	size int64
	at   time.Time
}

// New creates a catalog with empty ledger.
func New(cfg Config) (*Catalog, error) {
	for _, d := range []string{filepath.Join(cfg.Root, "staging"), filepath.Join(cfg.Root, "active")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	c := &Catalog{
		cfg:     cfg,
		entries: map[string]*Entry{},
		watcher: w,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
		subs:    map[*eventSubscription]struct{}{},
		tried:   map[string]probeOutcome{},
		diag:    map[string]*ObjectDiagnostic{},
	}
	if err := c.loadLedger(); err != nil {
		return nil, err
	}
	return c, nil
}

func key(id, version string) string { return id + "\x00" + version }

// Start scans both sources once and watches for changes.
func (c *Catalog) Start() error {
	dirs := c.sourceDirs()
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
		if err := c.watcher.Add(d); err != nil {
			return err
		}
		if err := c.scanDir(d); err != nil {
			return err
		}
	}
	c.pruneMissingSources()
	c.startMu.Lock()
	c.running = true
	c.startMu.Unlock()
	go c.loop()
	return nil
}

// Stop terminates watching. It is safe after a failed Start.
func (c *Catalog) Stop() {
	c.stopOnce.Do(func() { close(c.stop) })
	_ = c.watcher.Close()
	c.startMu.Lock()
	running := c.running
	c.startMu.Unlock()
	if running {
		<-c.done
	}
}

func (c *Catalog) sourceDirs() []string {
	var dirs []string
	if c.cfg.EssEnabled && c.cfg.EssDir != "" {
		dirs = append(dirs, c.cfg.EssDir)
	}
	if c.cfg.InboxEnabled && c.cfg.InboxDir != "" {
		dirs = append(dirs, c.cfg.InboxDir)
	}
	return dirs
}

func (c *Catalog) ensureWatchedDirs() {
	for _, dir := range c.sourceDirs() {
		if err := os.MkdirAll(dir, 0o755); err == nil {
			_ = c.watcher.Add(dir)
		}
	}
}

func (c *Catalog) pruneMissingSources() {
	c.mu.RLock()
	paths := make([]string, 0)
	seen := map[string]struct{}{}
	for _, e := range c.entries {
		p := e.Manifest.Artifact.SourcePath
		if p == "" {
			continue
		}
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		if _, err := os.Stat(p); os.IsNotExist(err) {
			paths = append(paths, p)
		}
	}
	c.mu.RUnlock()
	for _, p := range paths {
		c.handleRemoval(p)
	}
}

// Subscribe returns a buffered event channel; unsubscribe via the returned
// release function.
func (c *Catalog) Subscribe() (<-chan Event, func()) {
	sub := &eventSubscription{ch: make(chan Event, 16), done: make(chan struct{})}
	c.subMu.Lock()
	c.subs[sub] = struct{}{}
	c.subMu.Unlock()
	return sub.ch, func() {
		c.subMu.Lock()
		if _, ok := c.subs[sub]; ok {
			delete(c.subs, sub)
			close(sub.done)
		}
		c.subMu.Unlock()
	}
}

// Lookup resolves a model id and optional version ("" = newest installed).
func (c *Catalog) Lookup(id, version string) (*Entry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if version != "" {
		e, ok := c.entries[key(id, version)]
		return cloneEntry(e), ok
	}
	// Floating resolution: only portable (non-platform) packages are
	// eligible; compiled, platform-specific packages must be pinned by
	// exact version. Among portable packages the highest base semantic
	// version wins; ties prefer the declared higher precision (FP32 >
	// FP16 > INT8), then the numerically compared revision.
	var best *Entry
	for _, e := range c.entries {
		if e.ID != id {
			continue
		}
		base, rank, platform := parseProfileSuffix(e.Version)
		if platform != "" {
			continue
		}
		if best == nil {
			best = e
			continue
		}
		bestBase, bestRank, _ := parseProfileSuffix(best.Version)
		if cmp := compareVersions(base, bestBase); cmp != 0 {
			if cmp > 0 {
				best = e
			}
			continue
		}
		if rank != bestRank {
			if rank > bestRank {
				best = e
			}
			continue
		}
		if compareRevision(e.Revision, best.Revision) > 0 {
			best = e
		}
	}
	return cloneEntry(best), best != nil
}

// precisionSuffixRe recognizes the v1 profile-in-version grammar:
// <base-semver>-<precision>[-<platform>] (contract v0.3 §8), e.g.
// "1.0.0-fp16", "1.0.0-int8", "1.0.0-int8-rk3588". The optional platform
// token marks a compiled, device-family-bound artifact (ecosystem B).
var precisionSuffixRe = regexp.MustCompile(
	`^(.*)-(fp32|fp16|int8)(?:-([a-z0-9]+(?:[._-][a-z0-9]+)*))?$`)

// parseProfileSuffix returns the base version, precision rank (fp32=3,
// fp16=2, int8=1, unmarked=0) and platform token ("" for portable).
func parseProfileSuffix(version string) (base string, rank int, platform string) {
	m := precisionSuffixRe.FindStringSubmatch(version)
	if m == nil {
		return version, 0, ""
	}
	switch m[2] {
	case "fp32":
		rank = 3
	case "fp16":
		rank = 2
	case "int8":
		rank = 1
	}
	return m[1], rank, m[3]
}

func cloneEntry(e *Entry) *Entry {
	if e == nil {
		return nil
	}
	cp := *e
	return &cp
}

func (c *Catalog) loop() {
	defer close(c.done)
	const debounce = 300 * time.Millisecond
	pending := map[string]struct{}{}
	var flush <-chan time.Time
	for {
		select {
		case <-c.stop:
			return
		case ev, ok := <-c.watcher.Events:
			if !ok {
				return
			}
			if ev.Op&(fsnotify.Create|fsnotify.Write|fsnotify.Rename|fsnotify.Remove) == 0 {
				continue
			}
			pending[ev.Name] = struct{}{}
			if flush == nil {
				flush = time.After(debounce)
			}
		case <-flush:
			c.ensureWatchedDirs()
			for p := range pending {
				c.reconcilePath(p)
			}
			pending = map[string]struct{}{}
			flush = nil
		case _, ok := <-c.watcher.Errors:
			if !ok {
				return
			}
			// inotify overflow or transient watch loss: rebuild watches and
			// rescan so add/remove events cannot be lost permanently.
			c.ensureWatchedDirs()
			for _, dir := range c.sourceDirs() {
				_ = c.scanDir(dir)
			}
		}
	}
}

func (c *Catalog) reconcilePath(p string) {
	if _, err := os.Stat(p); os.IsNotExist(err) {
		c.handleRemoval(p)
		return
	}
	_ = c.ingestFile(p)
}

func (c *Catalog) scanDir(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		_ = c.ingestFile(filepath.Join(dir, e.Name()))
	}
	return nil
}

func (c *Catalog) handleRemoval(path string) {
	c.mu.Lock()
	var removed []Event
	var records []ledgerRecord
	var dirs []string
	detached := make([]*Entry, 0)
	for k, e := range c.entries {
		if e.Manifest == nil || e.Manifest.Artifact.SourcePath != path {
			continue
		}
		removed = append(removed, Event{Kind: EventRemoved, ModelID: e.ID, Version: e.Version})
		records = append(records, ledgerRecord{
			ObjectID:    filepath.Base(path),
			ID:          e.ID,
			Version:     e.Version,
			Revision:    e.Revision,
			State:       model.StateRemoved,
			SourcePath:  path,
			InstalledAt: time.Now().UTC(),
		})
		dirs = append(dirs, e.Dir)
		detached = append(detached, e)
		delete(c.entries, k)
	}
	c.ledger.Records = append(c.ledger.Records, records...)
	c.mu.Unlock()
	if c.cfg.BeforeRemove != nil {
		for _, e := range detached {
			c.cfg.BeforeRemove(e.ID, e.Version)
		}
	}
	for _, dir := range dirs {
		_ = os.RemoveAll(dir)
		// Remove the now-empty active/<id> parent when its last version is
		// gone; Remove is a no-op when the directory still holds versions.
		_ = os.Remove(filepath.Dir(dir))
	}
	if len(removed) > 0 {
		_ = c.saveLedger()
	}
	for _, ev := range removed {
		c.publish(ev)
	}
}

func (c *Catalog) ingestFile(path string) error {
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return err
	}
	if c.alreadyProbed(path, st.Size(), st.ModTime()) {
		return nil
	}
	tmp, err := os.MkdirTemp(filepath.Join(c.cfg.Root, "staging"), "pkg-*.tmp")
	if err != nil {
		return err
	}
	ok, err := extractPackage(path, tmp)
	if err != nil {
		_ = os.RemoveAll(tmp)
		c.rejectObject(path, st.Size(), err)
		c.markProbe(path, st.Size(), st.ModTime(), false)
		return err
	}
	if !ok {
		_ = os.RemoveAll(tmp)
		c.ignoreObject(path, st.Size(), "not a gzip Nereid model package containing root nereid-model.json")
		c.markProbe(path, st.Size(), st.ModTime(), false)
		return nil
	}
	m, labels, err := model.Load(tmp)
	if err != nil {
		_ = os.RemoveAll(tmp)
		c.rejectObject(path, st.Size(), err)
		c.markProbe(path, st.Size(), st.ModTime(), false)
		return err
	}
	if err := verifyFiles(tmp, m.Artifact.Files); err != nil {
		_ = os.RemoveAll(tmp)
		err = fmt.Errorf("verify %s: %w", path, err)
		c.rejectObject(path, st.Size(), err)
		c.markProbe(path, st.Size(), st.ModTime(), false)
		return err
	}
	if !model.IsSafeName(m.Model.ID) || !model.IsSafeName(m.Model.Version) {
		_ = os.RemoveAll(tmp)
		err = fmt.Errorf("unsafe model id/version in package %s", path)
		c.rejectObject(path, st.Size(), err)
		c.markProbe(path, st.Size(), st.ModTime(), false)
		return err
	}
	target := filepath.Join(c.cfg.Root, "active", m.Model.ID, m.Model.Version)
	if err := atomicInstall(tmp, target); err != nil {
		_ = os.RemoveAll(tmp)
		return err
	}
	m.Artifact.SourcePath = path
	entry := &Entry{
		ID: m.Model.ID, Version: m.Model.Version, Revision: m.Model.Revision,
		Dir: target, Manifest: m, Labels: labels,
	}
	c.mu.Lock()
	c.entries[key(m.Model.ID, m.Model.Version)] = entry
	c.ledger.Records = append(c.ledger.Records, ledgerRecord{
		ObjectID: filepath.Base(path), ID: m.Model.ID, Version: m.Model.Version,
		Revision: m.Model.Revision, State: model.StateReady, InstalledAt: time.Now().UTC(),
		SourcePath: path, SourceSize: st.Size(), SourceModAt: st.ModTime(),
	})
	c.mu.Unlock()
	c.markProbe(path, st.Size(), st.ModTime(), true)
	c.clearObjectDiagnostic(path)
	c.logf().Infof("accepted model object %s as %s/%s (format=%s)", filepath.Base(path), m.Model.ID, m.Model.Version, m.Artifact.Format)
	_ = c.saveLedger()
	c.publish(Event{Kind: EventAdded, ModelID: m.Model.ID, Version: m.Model.Version})
	return nil
}

func (c *Catalog) publish(ev Event) {
	c.subMu.Lock()
	subs := make([]*eventSubscription, 0, len(c.subs))
	for sub := range c.subs {
		subs = append(subs, sub)
	}
	c.subMu.Unlock()
	for _, sub := range subs {
		// Added events can be coalesced. Removal must not be silently
		// lost while a job exists, or it could serve an unlinked active
		// directory. Unsubscribe or shutdown are the only escape valves.
		if ev.Kind == EventRemoved {
			select {
			case sub.ch <- ev:
			case <-c.stop:
			case <-sub.done:
			}
			continue
		}
		select {
		case sub.ch <- ev:
		default:
		}
	}
}

// alreadyProbe skips reopening files whose size/mtime did not change since
// the last failed probe.
func (c *Catalog) alreadyProbed(path string, size int64, mtime time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.tried[path]
	// Failed probes may be corrected/replaced; retry them. Successful
	// objects are skipped only when both size and mtime are unchanged.
	return ok && p.ok && p.size == size && p.at.Equal(mtime)
}

func (c *Catalog) markProbe(path string, size int64, mtime time.Time, ok bool) {
	c.mu.Lock()
	c.tried[path] = probeOutcome{ok: ok, size: size, at: mtime}
	c.mu.Unlock()
}

func (c *Catalog) ignoreObject(path string, size int64, reason string) {
	c.recordObject(path, size, model.StateIgnored, reason)
}

func (c *Catalog) rejectObject(path string, size int64, cause error) {
	c.recordObject(path, size, model.StateFailed, cause.Error())
}

func (c *Catalog) recordObject(path string, size int64, state, reason string) {
	now := time.Now().UTC()
	c.diagMu.Lock()
	if c.diag == nil {
		c.diag = map[string]*ObjectDiagnostic{}
	}
	d := c.diag[path]
	if d == nil {
		d = &ObjectDiagnostic{Name: filepath.Base(path)}
		c.diag[path] = d
	}
	d.State = state
	d.Reason = reason
	d.SizeBytes = size
	d.Attempts++
	d.LastSeen = now
	c.diagMu.Unlock()

	if state == model.StateIgnored {
		c.logf().Debugf("ignored model-source object %s: %s", filepath.Base(path), reason)
	} else {
		c.logf().Errorf("rejected model-source object %s: %s", filepath.Base(path), reason)
	}
}

func (c *Catalog) clearObjectDiagnostic(path string) {
	c.diagMu.Lock()
	delete(c.diag, path)
	c.diagMu.Unlock()
}

// ObjectDiagnostics returns the most recent objects that were ignored or
// rejected. It is bounded so unrelated ESS traffic cannot grow memory without
// limit.
func (c *Catalog) ObjectDiagnostics() ObjectDiagnostics {
	c.diagMu.Lock()
	objects := make([]ObjectDiagnostic, 0, len(c.diag))
	out := ObjectDiagnostics{}
	for _, d := range c.diag {
		objects = append(objects, *d)
		switch d.State {
		case model.StateIgnored:
			out.IgnoredCount++
		default:
			out.RejectedCount++
		}
	}
	c.diagMu.Unlock()
	sort.Slice(objects, func(i, j int) bool {
		if objects[i].LastSeen.Equal(objects[j].LastSeen) {
			return objects[i].Name < objects[j].Name
		}
		return objects[i].LastSeen.After(objects[j].LastSeen)
	})
	if len(objects) > maxObjectDiagnostics {
		objects = objects[:maxObjectDiagnostics]
	}
	out.Objects = objects
	return out
}

type noopLogger struct{}

func (noopLogger) Debugf(string, ...any) {}
func (noopLogger) Infof(string, ...any)  {}
func (noopLogger) Errorf(string, ...any) {}

func (c *Catalog) logf() Logger {
	if c.cfg.Log != nil {
		return c.cfg.Log
	}
	return noopLogger{}
}

func atomicInstall(staging, target string) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	parent := filepath.Dir(target)
	finalTmp, err := os.MkdirTemp(parent, ".install-*")
	if err != nil {
		return err
	}
	_ = os.RemoveAll(finalTmp)
	if err := os.Rename(staging, finalTmp); err != nil {
		return err
	}
	if err := os.Chmod(finalTmp, 0o755); err != nil {
		return err
	}

	_, statErr := os.Stat(target)
	oldExists := statErr == nil
	var oldTmp string
	if oldExists {
		old, err := os.MkdirTemp(parent, ".replace-*")
		if err != nil {
			return err
		}
		_ = os.RemoveAll(old)
		oldTmp = old
		if err := os.Rename(target, oldTmp); err != nil {
			return err
		}
	}
	if err := os.Rename(finalTmp, target); err != nil {
		if oldTmp != "" {
			_ = os.Rename(oldTmp, target) // best-effort rollback
		}
		return err
	}
	if oldTmp != "" {
		_ = os.RemoveAll(oldTmp)
	}
	return nil
}

// verifyFiles enforces the package integrity contract: every declared entry
// must exist, be a regular file, have a positive exact byte count and a
// mandatory SHA-256 that matches; every other regular file in the package
// (except the root descriptor itself) must be declared. Full coverage is
// required so an undeclared payload cannot silently bypass verification.
func verifyFiles(dir string, files []model.FileEntry) error {
	declared := make(map[string]bool, len(files))
	for _, f := range files {
		clean, err := safePayloadRel(f.Path)
		if err != nil {
			return fmt.Errorf("unsafe payload path %q: %w", f.Path, err)
		}
		if clean == "nereid-model.json" {
			return fmt.Errorf("the root descriptor must not be listed in artifact.files")
		}
		if declared[clean] {
			return fmt.Errorf("payload %q is listed more than once", f.Path)
		}
		declared[clean] = true
		if f.Bytes <= 0 {
			return fmt.Errorf("payload %q must declare a positive byte count", f.Path)
		}
		if len(f.SHA256) != 64 {
			return fmt.Errorf("payload %q must declare a SHA-256 digest", f.Path)
		}
		p := filepath.Join(dir, clean)
		st, err := os.Stat(p)
		if err != nil {
			return err
		}
		if !st.Mode().IsRegular() {
			return fmt.Errorf("payload %q must be a regular file", f.Path)
		}
		if st.Size() != f.Bytes {
			return fmt.Errorf("%s size mismatch: %d != %d", f.Path, st.Size(), f.Bytes)
		}
		h := sha256.New()
		fh, err := os.Open(p)
		if err != nil {
			return err
		}
		if _, err := io.Copy(h, fh); err != nil {
			fh.Close()
			return err
		}
		fh.Close()
		if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
			return fmt.Errorf("%s sha256 mismatch", f.Path)
		}
	}
	return filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		name := filepath.ToSlash(rel)
		if name == "nereid-model.json" {
			return nil
		}
		if !declared[name] {
			return fmt.Errorf("package file %q is not listed in artifact.files", name)
		}
		return nil
	})
}

func safePayloadRel(p string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(p))
	if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes package directory", p)
	}
	return filepath.ToSlash(clean), nil
}

// SortedEntries lists installed models newest first (diagnostics).
func (c *Catalog) SortedEntries() []Entry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Entry, 0, len(c.entries))
	for _, e := range c.entries {
		out = append(out, *e)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		return compareVersions(out[i].Version, out[j].Version) > 0
	})
	return out
}

func (c *Catalog) loadLedger() error {
	p := filepath.Join(c.cfg.Root, "ledger.json")
	b, err := os.ReadFile(p)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &c.ledger); err != nil {
		return err
	}
	// Reconstruct the latest READY entry for each model version; the ledger
	// itself is immutable audit history.
	ready := map[string]ledgerRecord{}
	for _, r := range c.ledger.Records {
		if r.State == model.StateReady {
			ready[key(r.ID, r.Version)] = r
		}
	}
	for _, r := range ready {
		if !model.IsSafeName(r.ID) || !model.IsSafeName(r.Version) {
			continue
		}
		dir := filepath.Join(c.cfg.Root, "active", r.ID, r.Version)
		m, labels, err := model.Load(dir)
		if err != nil {
			continue
		}
		m.Artifact.SourcePath = r.SourcePath
		c.entries[key(r.ID, r.Version)] = &Entry{
			ID: r.ID, Version: r.Version, Revision: r.Revision,
			Dir: dir, Manifest: m, Labels: labels,
		}
		if r.SourcePath != "" {
			if st, err := os.Stat(r.SourcePath); err == nil && r.SourceSize == st.Size() &&
				r.SourceModAt.Equal(st.ModTime()) {
				c.tried[r.SourcePath] = probeOutcome{ok: true, size: st.Size(), at: st.ModTime()}
			}
		}
	}
	return nil
}

func (c *Catalog) saveLedger() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	b, err := json.MarshalIndent(&c.ledger, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(c.cfg.Root, "ledger.json.tmp")
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(c.cfg.Root, "ledger.json"))
}
