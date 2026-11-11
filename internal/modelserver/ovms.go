// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package modelserver owns node-local runtime configuration for inference
// model servers behind a family-neutral ServableManager boundary. Managed
// adapters (OVMS, Triton) turn active inference bindings into server
// configuration and load/unload calls, and the external registry tracks
// servers this service cannot administrate. A managed adapter unloads a model
// as soon as either its binding or its distributed package disappears.
package modelserver

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
)

const (
	// ovmsFamily identifies an OpenVINO Model Server runtime in the descriptor.
	ovmsFamily = FamilyOVMS
	// configRelPath is the DS-owned OVMS configuration below ModelRoot.
	configRelPath = "runtime/ovms/config.json"
	// defaultReloadTimeout bounds one OVMS reload notification.
	defaultReloadTimeout = 5 * time.Second
)

var deviceRe = regexp.MustCompile(`^[A-Za-z0-9_.:,-]+$`)

// Config configures the node-local OVMS reconciler.
type Config struct {
	// ModelRoot is the DS host path of modelcatalog's root.
	ModelRoot string
	// ContainerRoot is the same root as mounted inside the OVMS container.
	ContainerRoot string
	// ReloadEndpoint is OVMS REST base URL, or empty to disable notifications.
	ReloadEndpoint string
	// GRPCEndpoint is the KServe gRPC endpoint carried by every OVMS lease.
	// Managed jobs must use the lease (or an explicit job override), never an
	// implicit backend fallback disconnected from the acquired servable.
	GRPCEndpoint string
	// Catalog supplies the authoritative installed package entries.
	Catalog *modelcatalog.Catalog
	// ReloadTimeout defaults to five seconds when zero.
	ReloadTimeout time.Duration
}

// Key identifies one OVMS servable created for one binding target.
type Key struct {
	ModelID      string
	ModelVersion string
	TargetDevice string
}

// Manager reconciles OVMS model_config_list from active bindings.
type Manager struct {
	cfg Config

	applyMu sync.Mutex
	mu      sync.Mutex
	refs    map[Key]int
	gens    map[Key]uint64

	appliedHash  string
	reloadFailed bool

	startOnce sync.Once
	stopOnce  sync.Once
	stop      chan struct{}
	done      chan struct{}
}

// ovmsLease represents one job binding's reference to a configured OVMS
// servable.
type ovmsLease struct {
	manager *Manager
	key     Key
	name    string
	gen     uint64
	once    sync.Once
}

// Family returns the descriptor token owned by this manager.
func (m *Manager) Family() string { return ovmsFamily }

type ovmsConfigFile struct {
	ModelConfigList []ovmsModel `json:"model_config_list"`
}

type ovmsModel struct {
	Config map[string]any `json:"config"`
}

// New creates the runtime configuration directory and installs an initial
// empty OVMS model list so OVMS can start before any model is distributed.
func New(cfg Config) (*Manager, error) {
	if cfg.Catalog == nil {
		return nil, fmt.Errorf("modelserver catalog is required")
	}
	if cfg.ModelRoot == "" {
		return nil, fmt.Errorf("modelserver model root is required")
	}
	if strings.TrimSpace(cfg.GRPCEndpoint) == "" {
		return nil, fmt.Errorf("modelserver OVMS gRPC endpoint is required")
	}
	if cfg.ContainerRoot == "" {
		cfg.ContainerRoot = "/models"
	}
	if cfg.ReloadTimeout <= 0 {
		cfg.ReloadTimeout = defaultReloadTimeout
	}
	cfg.ModelRoot = filepath.Clean(cfg.ModelRoot)
	cfg.ContainerRoot = path.Clean("/" + strings.Trim(cfg.ContainerRoot, "/"))
	if err := os.MkdirAll(filepath.Join(cfg.ModelRoot, filepath.Dir(configRelPath)), 0o755); err != nil {
		return nil, err
	}
	m := &Manager{
		cfg:  cfg,
		refs: map[Key]int{},
		gens: map[Key]uint64{},
		stop: make(chan struct{}),
		done: make(chan struct{}),
	}
	if err := m.writeInitialConfig(); err != nil {
		return nil, err
	}
	return m, nil
}

// Start listens for package add/remove events and removes orphan servables.
func (m *Manager) Start() {
	m.startOnce.Do(func() {
		events, unsubscribe := m.cfg.Catalog.Subscribe()
		go m.run(events, unsubscribe)
	})
}

// Stop terminates the catalog watcher. It does not remove active bindings.
func (m *Manager) Stop() {
	m.startOnce.Do(func() {
		// Allow Stop before Start without leaking a subscription.
		close(m.done)
	})
	m.stopOnce.Do(func() { close(m.stop) })
	<-m.done
}

// Acquire registers one binding's OVMS target and ensures the servable is
// present in the generated configuration. The lease must be released when the
// job stops or resolves to a different model version.
func (m *Manager) Acquire(_ context.Context, entry *modelcatalog.Entry, device string) (Lease, error) {
	if entry == nil || entry.Manifest == nil {
		return nil, fmt.Errorf("model entry is required")
	}
	if entry.Manifest.Runtime.Server.Family != ovmsFamily {
		return nil, fmt.Errorf("model %s/%s is not an OVMS package", entry.ID, entry.Version)
	}
	if err := validateOVMSArtifact(entry); err != nil {
		return nil, err
	}
	target, err := NormalizeTargetDevice(device)
	if err != nil {
		return nil, err
	}
	key := Key{ModelID: entry.ID, ModelVersion: entry.Version, TargetDevice: target}
	m.mu.Lock()
	m.refs[key]++
	gen := m.gens[key]
	m.mu.Unlock()

	_ = m.apply()
	return &ovmsLease{
		manager: m,
		key:     key,
		name:    ServedName(entry.ID, entry.Version, target),
		gen:     gen,
	}, nil
}

// Release removes one binding reference. It is safe to call more than once.
func (l *ovmsLease) Release() {
	l.once.Do(func() {
		l.manager.release(l.key, l.gen)
	})
}

// ModelName returns the unique KServe model name exposed by OVMS.
func (l *ovmsLease) ModelName() string { return l.name }

// Endpoint returns the OVMS KServe gRPC endpoint configured for this manager.
func (l *ovmsLease) Endpoint() string { return l.manager.cfg.GRPCEndpoint }

// Ref returns the distributed package identity held by the lease.
func (l *ovmsLease) Ref() ModelRef {
	return ModelRef{ID: l.key.ModelID, Version: l.key.ModelVersion}
}

// Current reports whether the lease still matches the manager's active
// generation. A package removal (or any render that drops the servable)
// bumps the generation: Release then becomes a deliberate no-op and the
// holder must treat the lease as stale and Acquire a fresh one instead of
// polling a served name the manager no longer renders.
func (l *ovmsLease) Current() bool {
	l.manager.mu.Lock()
	defer l.manager.mu.Unlock()
	return l.manager.gens[l.key] == l.gen
}

// ServedName returns the deterministic, node-local model-server model name
// for an immutable package and binding target. It is shared by every managed
// family adapter. The numeric package repository directory remains
// package-local (normally /1); it is not encoded as a semantic version.
func ServedName(id, version, device string) string {
	target := strings.ToUpper(strings.TrimSpace(device))
	if target == "" {
		target = "CPU"
	}
	sum := sha256.Sum256([]byte(id + "\x00" + version + "\x00" + target))
	return "nereid_" + hex.EncodeToString(sum[:4]) + "_" + slug(id) + "_" + slug(version) + "_" + slug(target)
}

// NormalizeTargetDevice validates and normalizes an OVMS target-device
// selector. Empty selects CPU; OVMS compound selectors such as MULTI:GPU,CPU
// are permitted but arbitrary shell text is rejected.
func NormalizeTargetDevice(device string) (string, error) {
	target := strings.ToUpper(strings.TrimSpace(device))
	if target == "" {
		target = "CPU"
	}
	if len(target) > 64 || !deviceRe.MatchString(target) {
		return "", fmt.Errorf("invalid OVMS target device %q", device)
	}
	return target, nil
}

func (m *Manager) release(key Key, gen uint64) {
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

// RemoveModel detaches every target-device servable for one model version.
// It is used as catalog's pre-removal hook so OVMS unloads the model before
// its active files are deleted; a failed reload never blocks package revocation.
func (m *Manager) RemoveModel(id, version string) {
	m.mu.Lock()
	for key := range m.refs {
		if key.ModelID == id && key.ModelVersion == version {
			delete(m.refs, key)
			m.gens[key]++
		}
	}
	m.mu.Unlock()
	_ = m.apply()
}

func (m *Manager) run(events <-chan modelcatalog.Event, unsubscribe func()) {
	defer close(m.done)
	defer unsubscribe()
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-events:
			_ = m.apply()
		case <-ticker.C:
			m.mu.Lock()
			retry := m.reloadFailed && len(m.refs) > 0
			m.mu.Unlock()
			if retry {
				_ = m.apply()
			}
		}
	}
}

func (m *Manager) apply() error {
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

	cfg := ovmsConfigFile{ModelConfigList: []ovmsModel{}}
	seen := map[Key]struct{}{}
	for _, key := range keys {
		entry, ok := m.cfg.Catalog.Lookup(key.ModelID, key.ModelVersion)
		if !ok || entry.Manifest == nil || entry.Manifest.Runtime.Server.Family != ovmsFamily {
			continue
		}
		if err := validateOVMSArtifact(entry); err != nil {
			continue
		}
		base, err := m.basePath(entry)
		if err != nil {
			return err
		}
		cfg.ModelConfigList = append(cfg.ModelConfigList, ovmsModel{Config: map[string]any{
			"name":          ServedName(key.ModelID, key.ModelVersion, key.TargetDevice),
			"base_path":     base,
			"target_device": key.TargetDevice,
		}})
		seen[key] = struct{}{}
	}
	m.pruneStaleGenerations(seen)

	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	sum := sha256.Sum256(b)
	hash := hex.EncodeToString(sum[:])
	m.mu.Lock()
	unchanged := hash == m.appliedHash
	retryFailed := m.reloadFailed
	m.mu.Unlock()
	if !unchanged {
		if err := m.writeConfig(b); err != nil {
			return err
		}
		m.mu.Lock()
		m.appliedHash = hash
		m.mu.Unlock()
	}
	if unchanged && !retryFailed {
		return nil
	}
	if m.cfg.ReloadEndpoint == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), m.cfg.ReloadTimeout)
	defer cancel()
	if err := m.reload(ctx); err != nil {
		m.mu.Lock()
		m.reloadFailed = true
		m.mu.Unlock()
		return err
	}
	m.mu.Lock()
	m.reloadFailed = false
	m.mu.Unlock()
	return nil
}

func (m *Manager) pruneStaleGenerations(active map[Key]struct{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for key := range m.refs {
		if _, ok := active[key]; ok {
			continue
		}
		// The package was removed or is no longer an OVMS package. Drop its
		// old leases and invalidate any lease released after redistribution.
		delete(m.refs, key)
		m.gens[key]++
	}
}

// validateOVMSArtifact enforces the adapter-local repository contract. The
// catalog remains runtime-neutral; a Triton TensorRT package may be valid even
// though this OVMS adapter cannot serve it.
func validateOVMSArtifact(entry *modelcatalog.Entry) error {
	if entry == nil || entry.Manifest == nil {
		return fmt.Errorf("model entry is required")
	}
	m := entry.Manifest
	if m.Runtime.Server.Family != ovmsFamily {
		return fmt.Errorf("model %s/%s is not an OVMS package", entry.ID, entry.Version)
	}
	switch m.Artifact.Format {
	case model.ArtifactFormatONNX, model.ArtifactFormatOpenVINOIR:
	default:
		return fmt.Errorf("OVMS cannot serve artifact format %q for model %s/%s", m.Artifact.Format, entry.ID, entry.Version)
	}
	repository := strings.TrimSpace(m.Runtime.RepositoryPath)
	if repository == "" {
		repository = path.Join("repository", m.Runtime.ModelName)
	}
	clean, err := safeRelative(repository)
	if err != nil {
		return err
	}
	repoDir := filepath.Join(entry.Dir, filepath.FromSlash(clean))
	versions, err := os.ReadDir(repoDir)
	if err != nil {
		return fmt.Errorf("read OVMS repository for model %s/%s: %w", entry.ID, entry.Version, err)
	}
	// Contract v0.3 §8(§2): the NF package ships exactly one server version
	// directory named "1"; extra directories (numeric or otherwise) are
	// rejected so the served content is unambiguous.
	var dirs []string
	for _, version := range versions {
		if !version.IsDir() {
			continue
		}
		dirs = append(dirs, version.Name())
	}
	if len(dirs) != 1 || dirs[0] != "1" {
		return fmt.Errorf("OVMS repository %q must contain exactly one version directory named \"1\", got %v", clean, dirs)
	}
	versionDir := filepath.Join(repoDir, "1")
	switch m.Artifact.Format {
	case model.ArtifactFormatONNX:
		if err := requireNonEmptyRegularFiles(versionDir, "model.onnx"); err != nil {
			return err
		}
	case model.ArtifactFormatOpenVINOIR:
		if err := requireNonEmptyRegularFiles(versionDir, "model.xml", "model.bin"); err != nil {
			return err
		}
	}
	return nil
}

func requireNonEmptyRegularFiles(dir string, names ...string) error {
	for _, name := range names {
		p := filepath.Join(dir, name)
		st, err := os.Stat(p)
		if err != nil {
			return err
		}
		if !st.Mode().IsRegular() || st.Size() == 0 {
			return fmt.Errorf("OVMS payload %q must be a non-empty regular file", p)
		}
	}
	return nil
}

func (m *Manager) basePath(entry *modelcatalog.Entry) (string, error) {
	repository := strings.TrimSpace(entry.Manifest.Runtime.RepositoryPath)
	if repository == "" {
		repository = path.Join("repository", entry.Manifest.Runtime.ModelName)
	}
	clean, err := safeRelative(repository)
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(m.cfg.ModelRoot, filepath.Clean(entry.Dir))
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("model directory %q is outside model root %q", entry.Dir, m.cfg.ModelRoot)
	}
	return path.Join(m.cfg.ContainerRoot, filepath.ToSlash(relative), filepath.ToSlash(clean)), nil
}

func (m *Manager) writeInitialConfig() error {
	p := m.configPath()
	if _, err := os.Stat(p); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	b := []byte(`{"model_config_list":[]}` + "\n")
	if err := m.writeConfig(b); err != nil {
		return err
	}
	sum := sha256.Sum256(b)
	m.appliedHash = hex.EncodeToString(sum[:])
	return nil
}

func (m *Manager) writeConfig(b []byte) error {
	p := m.configPath()
	tmp, err := os.CreateTemp(filepath.Dir(p), ".config-*.json")
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
	return os.Rename(tmpName, p)
}

func (m *Manager) configPath() string {
	return filepath.Join(m.cfg.ModelRoot, filepath.FromSlash(configRelPath))
}

func (m *Manager) reload(ctx context.Context) error {
	endpoint := strings.TrimRight(m.cfg.ReloadEndpoint, "/") + "/v1/config/reload"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("OVMS config reload failed: %s", resp.Status)
	}
	return nil
}

func safeRelative(value string) (string, error) {
	clean := path.Clean("/" + strings.TrimSpace(value))
	if clean == "/" || clean == "" {
		return "", fmt.Errorf("empty repository path")
	}
	clean = strings.TrimPrefix(clean, "/")
	if clean == ".." || strings.HasPrefix(clean, "../") {
		return "", fmt.Errorf("unsafe repository path %q", value)
	}
	return clean, nil
}

func slug(value string) string {
	var b strings.Builder
	for _, r := range value {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	s := b.String()
	if len(s) > 32 {
		s = s[:32]
	}
	return s
}
