// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package modelserver

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

const (
	// tritonFamily identifies a managed Triton inference server.
	tritonFamily = FamilyTriton
	// tritonRepoRelPath is the DS-owned Triton model repository below
	// ModelRoot; the Triton container mounts it read-only.
	tritonRepoRelPath = "runtime/triton/repo"
	// tritonDefaultLoadTimeout bounds one repository load HTTP call.
	tritonDefaultLoadTimeout = 30 * time.Second
	// tritonDefaultReadyTimeout bounds readiness polling after a load.
	tritonDefaultReadyTimeout = 60 * time.Second
	// tritonReadyPollInterval is the ModelReady polling cadence during apply.
	tritonReadyPollInterval = 250 * time.Millisecond
	// tritonRetryInterval retries failed loads while leases remain active.
	tritonRetryInterval = 5 * time.Second
)

// tritonBackendByFormat maps package artifact formats to the Triton backend
// rendered into config.pbtxt when the descriptor does not carry an explicit
// backend hint. Compiled vendor engines (e.g. TensorRT plans) arrive through
// future artifact formats; ONNX load-time compilation remains the documented
// fallback, never an SDK dependency in this binary.
var tritonBackendByFormat = map[string]string{
	model.ArtifactFormatONNX:       "onnxruntime",
	model.ArtifactFormatOpenVINOIR: "openvino",
}

// tritonAllowedBackends bounds every backend string rendered into
// config.pbtxt; descriptor hints outside this allowlist are rejected rather
// than interpolated into server configuration text.
var tritonAllowedBackends = map[string]bool{
	"onnxruntime": true,
	"openvino":    true,
	"tensorrt":    true,
}

// tritonDeviceRe matches the Triton subset of target-device selectors:
// CPU, GPU or GPU.<index>. Compound server-specific selectors (e.g.
// MULTI:...) are rejected because instance_group needs an explicit kind.
var tritonDeviceRe = deviceRe

// TritonConfig configures the node-local managed Triton reconciler.
type TritonConfig struct {
	// ModelRoot is the DS host path of modelcatalog's root.
	ModelRoot string
	// ContainerRoot is the same root as mounted inside the Triton container.
	ContainerRoot string
	// ControlEndpoint is Triton's HTTP base URL (repository load/unload and
	// readiness), e.g. http://edgex-triton:8000.
	ControlEndpoint string
	// GRPCEndpoint is the OIP gRPC endpoint leases put in backend requests,
	// e.g. edgex-triton:8001.
	GRPCEndpoint string
	// Catalog supplies the authoritative installed package entries.
	Catalog *modelcatalog.Catalog
	// LoadTimeout defaults to tritonDefaultLoadTimeout when zero.
	LoadTimeout time.Duration
	// ReadyTimeout defaults to tritonDefaultReadyTimeout when zero.
	ReadyTimeout time.Duration
}

// TritonManager reconciles a Triton explicit-model-control repository from
// active bindings. It renders one served-name directory per
// (model, version, target device), drives load/unload through the Repository
// API and waits for ModelReady before a lease is used.
type TritonManager struct {
	cfg TritonConfig

	applyMu sync.Mutex
	mu      sync.Mutex
	refs    map[Key]int
	gens    map[Key]uint64
	failed  map[Key]bool

	httpClient *http.Client

	startOnce sync.Once
	stopOnce  sync.Once
	stop      chan struct{}
	done      chan struct{}
}

// tritonLease is one binding's reference to a rendered Triton servable.
type tritonLease struct {
	manager  *TritonManager
	key      Key
	name     string
	endpoint string
	gen      uint64
	once     sync.Once
}

// NewTriton creates the repository directory for a managed Triton server.
// No server call is made until the first Acquire.
func NewTriton(cfg TritonConfig) (*TritonManager, error) {
	if cfg.Catalog == nil {
		return nil, fmt.Errorf("triton modelserver catalog is required")
	}
	if cfg.ModelRoot == "" {
		return nil, fmt.Errorf("triton modelserver model root is required")
	}
	if cfg.ControlEndpoint == "" || cfg.GRPCEndpoint == "" {
		return nil, fmt.Errorf("triton control and gRPC endpoints are required")
	}
	if cfg.ContainerRoot == "" {
		cfg.ContainerRoot = "/models"
	}
	if cfg.LoadTimeout <= 0 {
		cfg.LoadTimeout = tritonDefaultLoadTimeout
	}
	if cfg.ReadyTimeout <= 0 {
		cfg.ReadyTimeout = tritonDefaultReadyTimeout
	}
	cfg.ModelRoot = filepath.Clean(cfg.ModelRoot)
	cfg.ContainerRoot = path.Clean("/" + strings.Trim(cfg.ContainerRoot, "/"))
	if err := os.MkdirAll(filepath.Join(cfg.ModelRoot, filepath.FromSlash(tritonRepoRelPath)), 0o755); err != nil {
		return nil, err
	}
	return &TritonManager{
		cfg:        cfg,
		refs:       map[Key]int{},
		gens:       map[Key]uint64{},
		failed:     map[Key]bool{},
		httpClient: &http.Client{Timeout: cfg.LoadTimeout},
		stop:       make(chan struct{}),
		done:       make(chan struct{}),
	}, nil
}

// Family returns the descriptor token owned by this manager.
func (m *TritonManager) Family() string { return tritonFamily }

// Start listens for package add/remove events and retries failed loads.
func (m *TritonManager) Start() {
	m.startOnce.Do(func() {
		events, unsubscribe := m.cfg.Catalog.Subscribe()
		go m.run(events, unsubscribe)
	})
}

// Stop terminates the catalog watcher. It does not unload active bindings.
func (m *TritonManager) Stop() {
	m.startOnce.Do(func() {
		close(m.done)
	})
	m.stopOnce.Do(func() { close(m.stop) })
	<-m.done
}

// Acquire renders and loads one Triton servable for the binding target. The
// lease must be released when the job stops or changes model version.
func (m *TritonManager) Acquire(_ context.Context, entry *modelcatalog.Entry, device string) (Lease, error) {
	if entry == nil || entry.Manifest == nil {
		return nil, fmt.Errorf("model entry is required")
	}
	if entry.Manifest.Runtime.Server.Family != tritonFamily {
		return nil, fmt.Errorf("model %s/%s is not a Triton package", entry.ID, entry.Version)
	}
	target, err := normalizeTritonDevice(device)
	if err != nil {
		return nil, err
	}
	if _, err := m.resolvePayload(entry, target); err != nil {
		return nil, err
	}
	key := Key{ModelID: entry.ID, ModelVersion: entry.Version, TargetDevice: target}
	m.mu.Lock()
	m.refs[key]++
	gen := m.gens[key]
	m.mu.Unlock()

	// A failed load keeps the binding reference and returns a lease: the
	// reconciler retries through its ticker while the job's readiness gate
	// surfaces the not-ready state. Only validation/configuration problems
	// reject the bind above this point.
	_ = m.apply()
	return &tritonLease{
		manager:  m,
		key:      key,
		name:     ServedName(entry.ID, entry.Version, target),
		endpoint: m.cfg.GRPCEndpoint,
		gen:      gen,
	}, nil
}

// Release removes one binding reference. It is safe to call more than once.
func (l *tritonLease) Release() {
	l.once.Do(func() { l.manager.release(l.key, l.gen) })
}

// ModelName returns the deterministic Triton model name for this lease.
func (l *tritonLease) ModelName() string { return l.name }

// Endpoint returns the Triton OIP gRPC endpoint backing the servable.
func (l *tritonLease) Endpoint() string { return l.endpoint }

// Ref returns the distributed package identity held by the lease.
func (l *tritonLease) Ref() ModelRef {
	return ModelRef{ID: l.key.ModelID, Version: l.key.ModelVersion}
}

// Current reports whether the lease generation still matches the manager.
func (l *tritonLease) Current() bool {
	l.manager.mu.Lock()
	defer l.manager.mu.Unlock()
	return l.manager.gens[l.key] == l.gen
}

// RemoveModel detaches every target-device servable of one model version
// before the catalog deletes its files. A failed unload never blocks package
// revocation; the retry loop reconciles the server when it returns.
func (m *TritonManager) RemoveModel(id, version string) {
	m.mu.Lock()
	for key := range m.refs {
		if key.ModelID == id && key.ModelVersion == version {
			delete(m.refs, key)
			delete(m.failed, key)
			m.gens[key]++
		}
	}
	m.mu.Unlock()
	_ = m.apply()
}

func (m *TritonManager) release(key Key, gen uint64) {
	m.mu.Lock()
	if m.gens[key] != gen {
		m.mu.Unlock()
		return
	}
	m.refs[key]--
	if m.refs[key] <= 0 {
		delete(m.refs, key)
	}
	m.mu.Unlock()
	_ = m.apply()
}

func (m *TritonManager) run(events <-chan modelcatalog.Event, unsubscribe func()) {
	defer close(m.done)
	defer unsubscribe()
	ticker := time.NewTicker(tritonRetryInterval)
	defer ticker.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-events:
			_ = m.apply()
		case <-ticker.C:
			m.mu.Lock()
			retry := len(m.failed) > 0 && len(m.refs) > 0
			m.mu.Unlock()
			if retry {
				_ = m.apply()
			}
		}
	}
}

// tritonRender is one desired served-name directory.
type tritonRender struct {
	key        Key
	name       string
	kind       string // KIND_CPU | KIND_GPU
	gpuIndex   int
	backend    string
	payloadVer string // absolute host path of the package "1" version directory
}

func (m *TritonManager) apply() error {
	m.applyMu.Lock()
	defer m.applyMu.Unlock()

	m.mu.Lock()
	keys := make([]Key, 0, len(m.refs))
	for key, count := range m.refs {
		if count > 0 {
			keys = append(keys, key)
		}
	}
	m.mu.Unlock()
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].ModelID != keys[j].ModelID {
			return keys[i].ModelID < keys[j].ModelID
		}
		if keys[i].ModelVersion != keys[j].ModelVersion {
			return keys[i].ModelVersion < keys[j].ModelVersion
		}
		return keys[i].TargetDevice < keys[j].TargetDevice
	})

	desired := map[Key]tritonRender{}
	var firstErr error
	for _, key := range keys {
		entry, ok := m.cfg.Catalog.Lookup(key.ModelID, key.ModelVersion)
		if !ok || entry.Manifest == nil || entry.Manifest.Runtime.Server.Family != tritonFamily {
			continue
		}
		render, err := m.buildRender(entry, key)
		if err != nil {
			m.markFailure(key, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		desired[key] = render
	}
	m.pruneStaleGenerations(desired)

	existing := m.existingRendered()
	// Load or reconcile desired servables first; removals only happen after
	// the new generation is ready, so a failed load never takes down the
	// previously served generation.
	for _, key := range keys {
		render, ok := desired[key]
		if !ok {
			continue
		}
		if _, present := existing[render.name]; present && !m.isFailed(key) {
			continue
		}
		if err := m.renderAndLoad(render); err != nil {
			m.markFailure(key, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		m.clearFailure(key)
		existing[render.name] = struct{}{}
	}

	// Unload servables that no longer have an active lease, then remove their
	// generated directories so the explicit-mode repository stays exact.
	wanted := map[string]struct{}{}
	for _, render := range desired {
		wanted[render.name] = struct{}{}
	}
	for name := range existing {
		if _, keep := wanted[name]; keep {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), m.cfg.LoadTimeout)
		status, err := m.repositoryPostStatus(ctx, name, "unload")
		cancel()
		// A 404 means the server no longer holds the model (failed load or
		// server restart); the end state the unload aims for already holds.
		if err != nil && status != http.StatusNotFound {
			// Keep the directory for the retry loop; deleting files behind a
			// server that still considers the model loaded is unsafe.
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		_ = os.RemoveAll(filepath.Join(m.repoRoot(), name))
	}
	return firstErr
}

func (m *TritonManager) renderAndLoad(render tritonRender) error {
	dir := filepath.Join(m.repoRoot(), render.name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := m.writeConfigPBTXT(dir, render); err != nil {
		return err
	}
	if err := m.linkVersion(dir, render.payloadVer); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), m.cfg.LoadTimeout)
	defer cancel()
	if err := m.repositoryPost(ctx, render.name, "load"); err != nil {
		return fmt.Errorf("triton load %s: %w", render.name, err)
	}
	readyCtx, readyCancel := context.WithTimeout(context.Background(), m.cfg.ReadyTimeout)
	defer readyCancel()
	if err := m.waitReady(readyCtx, render.name); err != nil {
		return fmt.Errorf("triton model %s not ready: %w", render.name, err)
	}
	return nil
}

func (m *TritonManager) writeConfigPBTXT(dir string, render tritonRender) error {
	var b strings.Builder
	fmt.Fprintf(&b, "name: %q\n", render.name)
	fmt.Fprintf(&b, "backend: %q\n", render.backend)
	// Current descriptor vision inputs are static [1,3,H,W]. Dynamic-batch
	// packages ([-1,3,S,S], NF served_by work) switch this to max_batch_size>0
	// plus dynamic_batching when that descriptor extension lands.
	b.WriteString("max_batch_size: 0\n")
	if render.kind == "KIND_GPU" {
		fmt.Fprintf(&b, "instance_group [ { kind: KIND_GPU, gpus: [ %d ] } ]\n", render.gpuIndex)
	} else {
		b.WriteString("instance_group [ { kind: KIND_CPU } ]\n")
	}
	p := filepath.Join(dir, "config.pbtxt")
	tmp, err := os.CreateTemp(dir, ".config-*.pbtxt")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.WriteString(b.String()); err != nil {
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
	return os.Rename(tmpName, p)
}

// linkVersion points <servedName>/1 at the package payload version directory
// through a relative symlink. Both trees live under the mounted ModelRoot, so
// the link resolves identically on the host and inside the Triton container;
// payload bytes are never copied or transformed.
func (m *TritonManager) linkVersion(dir, payloadVersionDir string) error {
	link := filepath.Join(dir, "1")
	target, err := filepath.EvalSymlinks(payloadVersionDir)
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(dir, target)
	if err != nil {
		return err
	}
	if current, err := os.Readlink(link); err == nil {
		if current == rel {
			return nil
		}
		_ = os.Remove(link)
	} else if !os.IsNotExist(err) {
		return err
	}
	return os.Symlink(rel, link)
}

func (m *TritonManager) repositoryPost(ctx context.Context, name, action string) error {
	_, err := m.repositoryPostStatus(ctx, name, action)
	return err
}

// repositoryPostStatus performs one Repository API call and returns the HTTP
// status alongside the error so callers can treat 404 on unload as the
// already-absent success state.
func (m *TritonManager) repositoryPostStatus(ctx context.Context, name, action string) (int, error) {
	endpoint := strings.TrimRight(m.cfg.ControlEndpoint, "/") +
		"/v2/repository/models/" + name + "/" + action
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
	if err != nil {
		return 0, err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, fmt.Errorf("repository %s failed: %s", action, resp.Status)
	}
	return resp.StatusCode, nil
}

func (m *TritonManager) waitReady(ctx context.Context, name string) error {
	endpoint := strings.TrimRight(m.cfg.ControlEndpoint, "/") + "/v2/models/" + name + "/ready"
	ticker := time.NewTicker(tritonReadyPollInterval)
	defer ticker.Stop()
	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return err
		}
		resp, err := m.httpClient.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1024))
			ready := resp.StatusCode == http.StatusOK
			_ = resp.Body.Close()
			if ready {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			if err != nil {
				return err
			}
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (m *TritonManager) buildRender(entry *modelcatalog.Entry, key Key) (tritonRender, error) {
	target, err := normalizeTritonDevice(key.TargetDevice)
	if err != nil {
		return tritonRender{}, err
	}
	payload, err := m.resolvePayload(entry, target)
	if err != nil {
		return tritonRender{}, err
	}
	kind, gpuIndex := "KIND_CPU", 0
	if idx, ok := tritonGPUSelector(target); ok {
		kind, gpuIndex = "KIND_GPU", idx
	}
	return tritonRender{
		key:        key,
		name:       ServedName(key.ModelID, key.ModelVersion, key.TargetDevice),
		kind:       kind,
		gpuIndex:   gpuIndex,
		backend:    payload.backend,
		payloadVer: payload.versionDir,
	}, nil
}

// tritonPayload is the resolved server-ready version directory for one
// served_by slot (or the legacy repository_path fallback).
type tritonPayload struct {
	backend    string
	versionDir string // host path of the directory containing exactly one "1"
}

func (m *TritonManager) resolvePayload(entry *modelcatalog.Entry, target string) (tritonPayload, error) {
	if err := validateTritonArtifact(entry); err != nil {
		return tritonPayload{}, err
	}
	backend := tritonBackendByFormat[entry.Manifest.Artifact.Format]
	repository := strings.TrimSpace(entry.Manifest.Runtime.RepositoryPath)
	for _, slot := range entry.Manifest.Runtime.ServedBy {
		if slot.Family != tritonFamily {
			continue
		}
		if len(slot.Targets) > 0 && !containsToken(slot.Targets, target) {
			continue
		}
		if slot.BackendHint != "" {
			hint := strings.ToLower(strings.TrimSpace(slot.BackendHint))
			if !tritonAllowedBackends[hint] {
				return tritonPayload{}, fmt.Errorf("triton backend hint %q is not allowed for model %s/%s", slot.BackendHint, entry.ID, entry.Version)
			}
			backend = hint
		}
		repository = strings.TrimSpace(slot.Payload)
		break
	}
	if repository == "" {
		repository = path.Join("repository", entry.Manifest.Runtime.ModelName)
	}
	clean, err := safeRelative(repository)
	if err != nil {
		return tritonPayload{}, err
	}
	repoDir := filepath.Join(entry.Dir, filepath.FromSlash(clean))
	if err := requireSingleVersionOne(repoDir); err != nil {
		return tritonPayload{}, err
	}
	versionDir := filepath.Join(repoDir, "1")
	switch entry.Manifest.Artifact.Format {
	case model.ArtifactFormatONNX:
		if err := requireNonEmptyRegularFiles(versionDir, "model.onnx"); err != nil {
			return tritonPayload{}, err
		}
	case model.ArtifactFormatOpenVINOIR:
		if err := requireNonEmptyRegularFiles(versionDir, "model.xml", "model.bin"); err != nil {
			return tritonPayload{}, err
		}
	}
	if !tritonAllowedBackends[backend] {
		return tritonPayload{}, fmt.Errorf("no triton backend mapping for artifact format %q on model %s/%s", entry.Manifest.Artifact.Format, entry.ID, entry.Version)
	}
	return tritonPayload{backend: backend, versionDir: versionDir}, nil
}

// validateTritonArtifact enforces the adapter-local package contract. The
// catalog stays runtime-neutral; an OVMS-only package may be valid even though
// this adapter cannot serve it.
func validateTritonArtifact(entry *modelcatalog.Entry) error {
	if entry == nil || entry.Manifest == nil {
		return fmt.Errorf("model entry is required")
	}
	manifest := entry.Manifest
	if manifest.Runtime.Server.Family != tritonFamily {
		return fmt.Errorf("model %s/%s is not a Triton package", entry.ID, entry.Version)
	}
	switch manifest.Artifact.Format {
	case model.ArtifactFormatONNX, model.ArtifactFormatOpenVINOIR:
	default:
		return fmt.Errorf("triton cannot serve artifact format %q for model %s/%s", manifest.Artifact.Format, entry.ID, entry.Version)
	}
	return nil
}

func (m *TritonManager) pruneStaleGenerations(active map[Key]tritonRender) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for key := range m.refs {
		if _, ok := active[key]; ok {
			continue
		}
		// The package was removed or is no longer a Triton package. Drop its
		// old leases and invalidate any lease released after redistribution.
		delete(m.refs, key)
		delete(m.failed, key)
		m.gens[key]++
	}
}

func (m *TritonManager) existingRendered() map[string]struct{} {
	entries, err := os.ReadDir(m.repoRoot())
	if err != nil {
		return map[string]struct{}{}
	}
	out := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			out[entry.Name()] = struct{}{}
		}
	}
	return out
}

func (m *TritonManager) repoRoot() string {
	return filepath.Join(m.cfg.ModelRoot, filepath.FromSlash(tritonRepoRelPath))
}

func (m *TritonManager) markFailure(key Key, err error) {
	m.mu.Lock()
	m.failed[key] = true
	m.mu.Unlock()
}

func (m *TritonManager) clearFailure(key Key) {
	m.mu.Lock()
	delete(m.failed, key)
	m.mu.Unlock()
}

func (m *TritonManager) isFailed(key Key) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.failed[key]
}

// normalizeTritonDevice validates the Triton subset of target-device
// selectors and returns the normalized key stored in the lease map.
func normalizeTritonDevice(device string) (string, error) {
	target := strings.ToUpper(strings.TrimSpace(device))
	if target == "" {
		target = "CPU"
	}
	if len(target) > 64 || !tritonDeviceRe.MatchString(target) {
		return "", fmt.Errorf("invalid triton target device %q", device)
	}
	if target == "CPU" || target == "GPU" {
		return target, nil
	}
	if strings.HasPrefix(target, "GPU.") {
		idx := strings.TrimPrefix(target, "GPU.")
		n, err := strconv.Atoi(idx)
		if err != nil || n < 0 {
			return "", fmt.Errorf("invalid triton GPU index in target device %q", device)
		}
		return target, nil
	}
	return "", fmt.Errorf("triton supports CPU, GPU or GPU.<index> target devices, got %q", device)
}

// tritonGPUSelector reports whether a normalized target selects a GPU and
// returns its instance_group device index.
func tritonGPUSelector(target string) (int, bool) {
	if target == "GPU" {
		return 0, true
	}
	if strings.HasPrefix(target, "GPU.") {
		if n, err := strconv.Atoi(strings.TrimPrefix(target, "GPU.")); err == nil && n >= 0 {
			return n, true
		}
	}
	return 0, false
}

func containsToken(tokens []string, want string) bool {
	for _, token := range tokens {
		if strings.EqualFold(strings.TrimSpace(token), want) {
			return true
		}
	}
	return false
}

func requireSingleVersionOne(repoDir string) error {
	versions, err := os.ReadDir(repoDir)
	if err != nil {
		return fmt.Errorf("read triton repository %q: %w", repoDir, err)
	}
	var dirs []string
	for _, version := range versions {
		if version.IsDir() {
			dirs = append(dirs, version.Name())
		}
	}
	if len(dirs) != 1 || dirs[0] != "1" {
		return fmt.Errorf("triton repository %q must contain exactly one version directory named \"1\", got %v", repoDir, dirs)
	}
	return nil
}
