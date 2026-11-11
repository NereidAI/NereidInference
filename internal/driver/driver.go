// -*- Mode: Go; indent-tabs-mode: t -*-
//
// Copyright (C) 2026 YIQISOFT
//
// SPDX-License-Identifier: Apache-2.0

// Package driver implements the EdgeX v4 ProtocolDriver contract for the
// generic inference device service. One EdgeX device maps to one inference
// pipeline: a media source bound to one descriptor-driven model package.
package driver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/edgexfoundry/device-sdk-go/v4/pkg/interfaces"
	dsModels "github.com/edgexfoundry/device-sdk-go/v4/pkg/models"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/clients/logger"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/common"
	"github.com/edgexfoundry/go-mod-core-contracts/v4/models"

	"github.com/edgexfoundry/device-inference/internal/config"
	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/job"
	"github.com/edgexfoundry/device-inference/internal/manager"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/publish"
)

// ProtocolName is the protocol properties key every inference device uses.
// It deliberately contains no vendor or server name.
const ProtocolName = "Inference"

const (
	// ManagerDeviceName is the singleton service-level device exposing model
	// inventory to northbound protocol bridges through ordinary core-command
	// GET commands.
	ManagerDeviceName = "nereid-inference"
	// ManagerProfileName is its read-only management profile.
	ManagerProfileName = "nereid-inference-manager"
	// ModelsResource is the full replacement model inventory JSON string.
	ModelsResource = "ModelsJSON"
	// ObjectDiagnosticsResource reports ignored/rejected distribution objects.
	ObjectDiagnosticsResource = "ObjectDiagnosticsJSON"
	// ExternalServablesResource is the read-only merged health view of inference
	// servers NI cannot administrate. Managed external delivery is unsupported.
	ExternalServablesResource = "ExternalServablesJSON"
	// PipelinesResource is the atomic full-table NI-native pipeline registry.
	PipelinesResource = "PipelinesJSON"
	// ActivePipelineVersionResource is the Executor.ResolvedVersion fact.
	ActivePipelineVersionResource = "ActivePipelineVersion"
	// LatestPipelineVersionResource is the Registry.Lookup newest-version fact.
	LatestPipelineVersionResource = "LatestPipelineVersion"
)

// InferenceDriver is the singleton ProtocolDriver handed to the SDK.
type InferenceDriver struct {
	lc      logger.LoggingClient
	asyncCh chan<- *dsModels.AsyncValues
	sdk     interfaces.DeviceServiceSDK

	cfgMu     sync.RWMutex
	cfg       config.InferenceConfig
	customCfg *config.ServiceConfig

	catalog           *modelcatalog.Catalog
	frames            *frame.Manager
	secretGetter      func(string) (map[string]string, error)
	backends          *infer.Registry
	decoders          *decode.Registry
	servables         *modelserver.Registry
	externalServables *modelserver.ExternalRegistry
	pipelines         *pipeline.Registry
	inventory         *manager.Inventory
	preview           *publish.Server
	previewWg         sync.WaitGroup

	jobsMu sync.Mutex
	jobs   map[string]*jobState
}

type jobState struct {
	name string
	cfg  job.Config
	job  *job.Job
}

var (
	once   sync.Once
	driver *InferenceDriver
)

// NewInferenceDriver returns the singleton driver instance.
func NewInferenceDriver() interfaces.ProtocolDriver {
	once.Do(func() {
		driver = &InferenceDriver{}
	})
	return driver
}

// Initialize loads configuration and starts the shared local services.
func (d *InferenceDriver) Initialize(sdk interfaces.DeviceServiceSDK) error {
	d.sdk = sdk
	d.lc = sdk.LoggingClient()
	d.asyncCh = sdk.AsyncValuesChannel()
	d.jobs = make(map[string]*jobState)
	d.customCfg = &config.ServiceConfig{Inference: config.New()}

	if err := sdk.LoadCustomConfig(d.customCfg, "Inference"); err != nil {
		return fmt.Errorf("load Inference custom config: %w", err)
	}
	if err := d.customCfg.Inference.Validate(); err != nil {
		return fmt.Errorf("invalid Inference custom config: %w", err)
	}
	d.cfg = d.customCfg.Inference

	// Keeper decodes updates into this private target; it is copied into
	// d.cfg under cfgMu so it never rewrites the live configuration map.
	watchTarget := &config.InferenceConfig{}
	if err := sdk.ListenForCustomConfigChanges(watchTarget, "Inference", func(raw any) {
		next, ok := raw.(*config.InferenceConfig)
		if !ok {
			d.lc.Error("Ignored invalid Inference configuration update")
			return
		}
		if err := next.Validate(); err != nil {
			d.lc.Errorf("Ignoring invalid Inference configuration update: %v", err)
			return
		}
		d.cfgMu.Lock()
		d.cfg = *next
		d.cfgMu.Unlock()
		if d.inventory != nil {
			d.inventory.SetEndpoints(next.DefaultEndpoint, next.OpenAIEndpoint)
		}
	}); err != nil {
		return fmt.Errorf("listen for Inference config changes: %w", err)
	}

	cfg := d.currentConfig()
	d.secretGetter = func(name string) (map[string]string, error) {
		if name == "" || sdk.SecretProvider() == nil {
			return nil, nil
		}
		return sdk.SecretProvider().GetSecret(name)
	}
	d.backends = infer.NewRegistry(cfg.OpenAIAPISecret, d.secretGetter)
	d.decoders = decode.NewRegistry()
	pipelineRegistry, err := pipeline.NewRegistry(pipeline.RegistryConfig{
		ModelRoot: cfg.ModelRoot, ProvisionFile: cfg.Pipelines.ProvisionFile,
	})
	if err != nil {
		return fmt.Errorf("create pipeline registry: %w", err)
	}
	d.pipelines = pipelineRegistry

	registry := modelserver.NewRegistry()
	catalog, err := modelcatalog.New(modelcatalog.Config{
		Root:         cfg.ModelRoot,
		EssEnabled:   cfg.Ess.Enabled,
		EssDir:       cfg.Ess.LocalDir,
		InboxEnabled: cfg.Inbox.Enabled,
		InboxDir:     cfg.Inbox.Dir,
		BeforeRemove: func(id, version string) {
			// Fan out to every managed family so each server unloads the
			// model before its active files are deleted.
			if d.servables != nil {
				d.servables.RemoveModel(id, version)
			}
		},
		Log: d.lc,
	})
	if err != nil {
		return fmt.Errorf("create model catalog: %w", err)
	}
	d.catalog = catalog
	if cfg.OVMS.Enabled {
		ovmsManager, err := modelserver.New(modelserver.Config{
			ModelRoot:      cfg.ModelRoot,
			ContainerRoot:  cfg.OVMS.ContainerRoot,
			ReloadEndpoint: cfg.OVMS.RESTEndpoint,
			GRPCEndpoint:   cfg.DefaultEndpoint,
			Catalog:        d.catalog,
			ReloadTimeout:  time.Duration(cfg.OVMS.ReloadTimeoutSec) * time.Second,
		})
		if err != nil {
			catalog.Stop()
			return fmt.Errorf("create OVMS model-server manager: %w", err)
		}
		if err := registry.Register(ovmsManager); err != nil {
			catalog.Stop()
			return fmt.Errorf("register OVMS model-server manager: %w", err)
		}
	}
	if cfg.Triton.Enabled {
		tritonManager, err := modelserver.NewTriton(modelserver.TritonConfig{
			ModelRoot:       cfg.ModelRoot,
			ContainerRoot:   cfg.Triton.ContainerRoot,
			ControlEndpoint: cfg.Triton.ControlEndpoint,
			GRPCEndpoint:    cfg.Triton.GRPCEndpoint,
			Catalog:         d.catalog,
			LoadTimeout:     time.Duration(cfg.Triton.LoadTimeoutSec) * time.Second,
			ReadyTimeout:    time.Duration(cfg.Triton.ReadyTimeoutSec) * time.Second,
		})
		if err != nil {
			catalog.Stop()
			return fmt.Errorf("create Triton model-server manager: %w", err)
		}
		if err := registry.Register(tritonManager); err != nil {
			catalog.Stop()
			return fmt.Errorf("register Triton model-server manager: %w", err)
		}
	}
	d.servables = registry
	if cfg.External.Enabled {
		externalRegistry, err := modelserver.NewExternal(modelserver.ExternalConfig{
			ModelRoot:     cfg.ModelRoot,
			ProvisionFile: cfg.External.ProvisionFile,
			Prober:        &kserveProber{backends: d.backends},
		})
		if err != nil {
			catalog.Stop()
			return fmt.Errorf("create external servable registry: %w", err)
		}
		d.externalServables = externalRegistry
	}
	if err := catalog.Start(); err != nil {
		catalog.Stop()
		return fmt.Errorf("start model catalog: %w", err)
	}
	d.servables.Start()
	if d.externalServables != nil {
		d.externalServables.Start()
	}

	d.frames, err = frame.NewManager(toFrameConfig(cfg.FFmpeg), d.lc)
	if err != nil {
		return fmt.Errorf("create frame manager: %w", err)
	}
	d.inventory = manager.New(manager.Config{
		Catalog:        d.catalog,
		Backends:       d.backends,
		KServeEndpoint: cfg.DefaultEndpoint,
		OpenAIEndpoint: cfg.OpenAIEndpoint,
		Servables:      d.servables,
	})
	d.inventory.Start()
	hub := publish.NewHub()
	if cfg.Preview.Enabled {
		server, err := publish.NewServer(cfg.Preview.Bind, hub)
		if err != nil {
			return fmt.Errorf("bind preview server %s: %w", cfg.Preview.Bind, err)
		}
		d.preview = server
		d.startPreview()
	}

	d.lc.Info("InferenceDriver.Initialize: generic inference pipeline initialized")
	return nil
}

// Start admits all provisioned devices after shared services are ready.
func (d *InferenceDriver) Start() error {
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	var admissionErrors []error
	for _, dev := range d.sdk.Devices() {
		if dev.Name == ManagerDeviceName || dev.ProfileName == ManagerProfileName {
			continue
		}
		if err := d.admitLocked(dev.Name, dev.Protocols[ProtocolName]); err != nil {
			admissionErrors = append(admissionErrors, fmt.Errorf("%s: %w", dev.Name, err))
			d.lc.Errorf("Device %s failed to start: %v", dev.Name, err)
		}
	}
	if len(admissionErrors) > 0 {
		return fmt.Errorf("%d inference device(s) failed configuration validation: %w", len(admissionErrors), errors.Join(admissionErrors...))
	}
	d.lc.Infof("InferenceDriver.Start: started %d inference job(s)", len(d.jobs))
	return nil
}

// Stop terminates jobs and releases all shared resources.
func (d *InferenceDriver) Stop(bool) error {
	d.jobsMu.Lock()
	for _, st := range d.jobs {
		st.job.Stop()
	}
	d.jobs = map[string]*jobState{}
	d.jobsMu.Unlock()

	if d.preview != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_ = d.preview.Shutdown(ctx)
		cancel()
		d.previewWg.Wait()
	}
	if d.inventory != nil {
		d.inventory.Stop()
	}
	if d.externalServables != nil {
		d.externalServables.Stop()
	}
	if d.servables != nil {
		d.servables.Stop()
	}
	if d.catalog != nil {
		d.catalog.Stop()
	}
	if d.frames != nil {
		d.frames.StopAll()
	}
	if d.backends != nil {
		d.backends.Close()
	}
	d.lc.Info("InferenceDriver.Stop: all inference pipelines stopped")
	return nil
}

// HandleReadCommands returns cached latest values; it never performs live
// inference. Continuous/periodic jobs publish async events, and writes invoke
// on-demand cycles.
func (d *InferenceDriver) HandleReadCommands(deviceName string, _ map[string]models.ProtocolProperties, reqs []dsModels.CommandRequest) ([]*dsModels.CommandValue, error) {
	if deviceName == ManagerDeviceName {
		return d.handleManagerRead(reqs)
	}
	st, err := d.getJob(deviceName)
	if err != nil {
		return nil, err
	}
	out := make([]*dsModels.CommandValue, 0, len(reqs))
	for _, req := range reqs {
		var value any
		var ok bool
		switch req.DeviceResourceName {
		case "State", "LastError", "SuccessTotal", "ErrorTotal", "P50Ms", "P95Ms":
			value, ok = st.job.StatsSnapshot()[req.DeviceResourceName]
		case ActivePipelineVersionResource, LatestPipelineVersionResource:
			active, latest := st.job.PipelineVersions()
			ok = true
			if req.DeviceResourceName == ActivePipelineVersionResource {
				value = active
			} else {
				value = latest
			}
		default:
			value, ok = st.job.LastValue(req.DeviceResourceName)
		}
		if !ok {
			value = emptyValue(req.Type)
		}
		cv, err := dsModels.NewCommandValue(req.DeviceResourceName, req.Type, normalizeValue(value, req.Type))
		if err != nil {
			return nil, fmt.Errorf("create reading %s: %w", req.DeviceResourceName, err)
		}
		out = append(out, cv)
	}
	return out, nil
}

// handleManagerRead serves the singleton northbound management device. Its
// inventory is computed on demand: catalog/package state comes from
// modelcatalog, READY has already been gated by runtime probes, and active
// bindings are the count of running Job devices resolved to each model.
func (d *InferenceDriver) handleManagerRead(reqs []dsModels.CommandRequest) ([]*dsModels.CommandValue, error) {
	out := make([]*dsModels.CommandValue, 0, len(reqs))
	for _, req := range reqs {
		var raw []byte
		var err error
		switch req.DeviceResourceName {
		case ModelsResource:
			raw, err = json.Marshal(d.inventory.Models(d.bindingCounts()))
		case ObjectDiagnosticsResource:
			raw, err = json.Marshal(d.inventory.ObjectDiagnostics())
		case ExternalServablesResource:
			if d.externalServables == nil {
				raw = []byte("[]")
			} else {
				raw, err = json.Marshal(d.externalServables.Snapshot())
			}
		case PipelinesResource:
			raw, err = json.Marshal(d.pipelines.Snapshot())
		default:
			cv, cvErr := dsModels.NewCommandValue(req.DeviceResourceName, req.Type, "")
			if cvErr != nil {
				return nil, fmt.Errorf("create reading %s: %w", req.DeviceResourceName, cvErr)
			}
			out = append(out, cv)
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("marshal %s: %w", req.DeviceResourceName, err)
		}
		cv, err := dsModels.NewCommandValue(req.DeviceResourceName, req.Type, string(raw))
		if err != nil {
			return nil, fmt.Errorf("create reading %s: %w", req.DeviceResourceName, err)
		}
		out = append(out, cv)
	}
	return out, nil
}

// handleManagerWrite accepts full-table replacements for writable manager
// resources. ExternalServablesJSON is preflighted across both requests and
// parameters so malformed, parameterless and mixed writes all fail before any
// pipeline mutation or payload conversion.
func (d *InferenceDriver) handleManagerWrite(reqs []dsModels.CommandRequest, params []*dsModels.CommandValue) error {
	for _, req := range reqs {
		if req.DeviceResourceName == ExternalServablesResource {
			return modelserver.ErrExternalServablesReadOnly
		}
	}
	for _, param := range params {
		if param != nil && param.DeviceResourceName == ExternalServablesResource {
			return modelserver.ErrExternalServablesReadOnly
		}
	}
	for i, p := range params {
		name := p.DeviceResourceName
		if i < len(reqs) {
			name = reqs[i].DeviceResourceName
		}
		value, err := p.StringValue()
		if err != nil {
			return fmt.Errorf("manager resource %s payload must be a JSON string: %w", name, err)
		}
		switch name {
		case PipelinesResource:
			records, err := pipeline.DecodeDefinitionsJSON([]byte(value))
			if err != nil {
				return fmt.Errorf("parse pipelines table: %w", err)
			}
			if err := d.pipelines.Replace(records); err != nil {
				return err
			}
		default:
			return fmt.Errorf("manager resource %q is not writable", name)
		}
	}
	return nil
}

// kserveProber adapts the shared infer backend registry to the external
// servable probe boundary. Only OIP gRPC servers are valid v1 records.
type kserveProber struct {
	backends *infer.Registry
}

func (p *kserveProber) Probe(ctx context.Context, protocol, endpoint, modelName, modelVersion string) error {
	if protocol != "kserve-grpc" {
		return fmt.Errorf("unsupported external protocol %q", protocol)
	}
	backend, err := p.backends.Get("kserve", endpoint)
	if err != nil {
		return err
	}
	if err := backend.Ready(ctx, modelName, modelVersion); err != nil {
		return err
	}
	// Metadata cross-check: the server must expose model metadata for the
	// same name it reported ready. Descriptor I/O comparison lands when jobs
	// consume external servable ids in the pipeline executor.
	ks, ok := backend.(*infer.KServeBackend)
	if !ok {
		return fmt.Errorf("external %s endpoint did not resolve to a KServe backend", protocol)
	}
	_, _, err = ks.Metadata(ctx, modelName, modelVersion)
	return err
}

// bindingCounts counts active Job devices by installed model id/version.
func (d *InferenceDriver) bindingCounts() map[string]int {
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	counts := make(map[string]int)
	for _, st := range d.jobs {
		if st.cfg.PipelineID != "" {
			for _, ref := range st.job.PipelineModelRefs() {
				if _, ok := d.catalog.Lookup(ref.ID, ref.Version); ok {
					counts[manager.BindingKey(ref.ID, ref.Version)]++
				}
			}
			continue
		}
		entry, ok := d.catalog.Lookup(st.cfg.ModelID, st.cfg.ModelVersion)
		if !ok {
			continue
		}
		counts[manager.BindingKey(entry.ID, entry.Version)]++
	}
	return counts
}

// HandleWriteCommands invokes one immediate inference cycle. The written
// Trigger value may carry a prompt for LLM/VLM devices. Manager-device writes
// replace the pipeline table; merged external health remains read-only.
func (d *InferenceDriver) HandleWriteCommands(deviceName string, _ map[string]models.ProtocolProperties, reqs []dsModels.CommandRequest, params []*dsModels.CommandValue) error {
	if deviceName == ManagerDeviceName {
		return d.handleManagerWrite(reqs, params)
	}
	st, err := d.getJob(deviceName)
	if err != nil {
		return err
	}
	prompt := ""
	for i, p := range params {
		name := p.DeviceResourceName
		if i >= len(reqs) {
			name = "Trigger"
		}
		if name != "Trigger" {
			continue
		}
		if v, err := p.StringValue(); err == nil {
			prompt = v
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = st.job.Invoke(ctx, prompt)
	return err
}

// AddDevice validates and admits a dynamically added device.
func (d *InferenceDriver) AddDevice(deviceName string, protocols map[string]models.ProtocolProperties, _ models.AdminState) error {
	d.lc.Debugf("Inference device added: %s", deviceName)
	if deviceName == ManagerDeviceName {
		return nil
	}
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	return d.admitLocked(deviceName, protocols[ProtocolName])
}

// UpdateDevice rebuilds the job when protocol properties change.
func (d *InferenceDriver) UpdateDevice(deviceName string, protocols map[string]models.ProtocolProperties, _ models.AdminState) error {
	d.lc.Debugf("Inference device updated: %s", deviceName)
	if deviceName == ManagerDeviceName {
		return nil
	}
	cfg, err := job.FromProperties(protocols[ProtocolName])
	if err != nil {
		return err
	}
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	if old := d.jobs[deviceName]; old != nil && equalJobConfig(old.cfg, cfg) {
		return nil
	}
	if old := d.jobs[deviceName]; old != nil {
		old.job.Stop()
		delete(d.jobs, deviceName)
		if d.preview != nil {
			// Drop stale overlay/snapshot while the replacement job warms up.
			d.preview.Hub().Remove(deviceName)
		}
	}
	return d.admitLocked(deviceName, protocols[ProtocolName])
}

// RemoveDevice stops the pipeline and releases its frame subscription.
func (d *InferenceDriver) RemoveDevice(deviceName string, _ map[string]models.ProtocolProperties) error {
	d.lc.Debugf("Inference device removed: %s", deviceName)
	if deviceName == ManagerDeviceName {
		return nil
	}
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	if st := d.jobs[deviceName]; st != nil {
		st.job.Stop()
		delete(d.jobs, deviceName)
	}
	if d.preview != nil {
		d.preview.Hub().Remove(deviceName)
	}
	return nil
}

// Discover is unused; cameras and inference jobs are explicitly provisioned.
func (d *InferenceDriver) Discover() error {
	return errors.New("driver's Discover function isn't implemented")
}

// ValidateDevice checks admission-time protocol properties.
func (d *InferenceDriver) ValidateDevice(device models.Device) error {
	if device.Name == ManagerDeviceName || device.ProfileName == ManagerProfileName {
		return nil
	}
	props, ok := device.Protocols[ProtocolName]
	if !ok {
		return errors.New("'" + ProtocolName + "' protocol properties not found for device " + device.Name)
	}
	cfg, err := job.FromProperties(props)
	if err != nil {
		return err
	}
	if cfg.PipelineID != "" {
		if d.pipelines == nil {
			return fmt.Errorf("pipeline registry is not initialized")
		}
		if _, ok := d.pipelines.Lookup(cfg.PipelineID); !ok {
			return fmt.Errorf("Inference.PipelineId %q is not registered", cfg.PipelineID)
		}
	}
	return nil
}

func (d *InferenceDriver) currentConfig() config.InferenceConfig {
	d.cfgMu.RLock()
	defer d.cfgMu.RUnlock()
	return d.cfg
}

// admitLocked creates and starts one job. Callers must hold jobsMu.
func (d *InferenceDriver) admitLocked(deviceName string, props map[string]any) error {
	cfg, err := job.FromProperties(props)
	if err != nil {
		return err
	}
	if _, exists := d.jobs[deviceName]; exists {
		return nil
	}
	if cfg.PipelineID != "" {
		if _, ok := d.pipelines.Lookup(cfg.PipelineID); !ok {
			return fmt.Errorf("Inference.PipelineId %q is not registered", cfg.PipelineID)
		}
	}
	c := d.currentConfig()
	if c.Limits.MaxJobs > 0 && len(d.jobs) >= c.Limits.MaxJobs {
		return fmt.Errorf("maximum inference job count %d reached", c.Limits.MaxJobs)
	}

	var federation *frame.Federation
	if cfg.SourceKind == "onvif-camera" {
		federation = frame.NewFederation(c.CoreMetadataURL, c.CoreCommandURL, d.secretGetter, frame.FederationConfig{
			NormalizeVirtualChannel: c.Onvif.NormalizeVirtualChannel,
			MainChannel:             c.Onvif.MainChannel,
			Log:                     d.lc,
		})
	}
	j := job.New(deviceName, cfg, job.Deps{
		Catalog:               d.catalog,
		Frames:                d.frames,
		Federation:            federation,
		GetSecret:             d.secretGetter,
		Backends:              d.backends,
		Decoders:              d.decoders,
		Servables:             d.servables,
		Pipelines:             d.pipelines,
		Preview:               d.previewHubOrNil(),
		Log:                   d.lc,
		Emitter:               &asyncEmitter{sdk: d.sdk, ch: d.asyncCh, lc: d.lc},
		DefaultKServeEndpoint: c.DefaultEndpoint,
		DefaultOpenAIEndpoint: c.OpenAIEndpoint,
		CoreCommandURL:        c.CoreCommandURL,
		JSONCap:               c.Limits.JSONPayloadCap,
		FrameTTLFactor:        c.Limits.FrameTTLFactor,
	})
	j.Start()
	d.jobs[deviceName] = &jobState{name: deviceName, cfg: cfg, job: j}
	return nil
}

func (d *InferenceDriver) getJob(name string) (*jobState, error) {
	d.jobsMu.Lock()
	defer d.jobsMu.Unlock()
	st, ok := d.jobs[name]
	if !ok {
		return nil, fmt.Errorf("inference job %q is not running", name)
	}
	return st, nil
}

func (d *InferenceDriver) startPreview() {
	d.previewWg.Add(1)
	go func() {
		defer d.previewWg.Done()
		if err := d.preview.Serve(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			d.lc.Errorf("preview server stopped: %v", err)
		}
	}()
}

func (d *InferenceDriver) previewHubOrNil() *publish.Hub {
	if d.preview == nil {
		return nil
	}
	return d.preview.Hub()
}

func toFrameConfig(in config.FFmpegConfig) frame.FFmpegConfig {
	return frame.FFmpegConfig{
		Bin:                    in.Bin,
		ProbeBin:               in.ProbeBin,
		Global:                 in.Global,
		ConnectTimeoutSec:      in.ConnectTimeoutSec,
		InputTemplates:         in.InputTemplates,
		OutputTemplate:         in.OutputTemplate,
		ExtraArgs:              in.ExtraArgs,
		AuthChallengeNormalize: in.AuthChallengeNormalize,
		DropUID:                in.DropUID,
		DropGID:                in.DropGID,
		VAAPIDevice:            in.VAAPIDevice,
	}
}

func equalJobConfig(a, b job.Config) bool {
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}

func emptyValue(valueType string) any {
	switch valueType {
	case common.ValueTypeString:
		return ""
	case common.ValueTypeBool:
		return false
	case common.ValueTypeInt32:
		return int32(0)
	case common.ValueTypeInt64:
		return int64(0)
	case common.ValueTypeFloat64:
		return float64(0)
	case common.ValueTypeBinary:
		return []byte{}
	default:
		return ""
	}
}

func normalizeValue(v any, valueType string) any {
	switch valueType {
	case common.ValueTypeInt32:
		return toInt32(v)
	case common.ValueTypeInt64:
		return toInt64(v)
	case common.ValueTypeFloat64:
		return toFloat64(v)
	case common.ValueTypeBool:
		return toBool(v)
	case common.ValueTypeBinary:
		if b, ok := v.([]byte); ok {
			return b
		}
		return []byte{}
	default:
		return fmt.Sprintf("%v", v)
	}
}

func toInt32(v any) int32 {
	switch t := v.(type) {
	case int:
		return clampInt32(int64(t))
	case int32:
		return t
	case int64:
		return clampInt32(t)
	case float64:
		return clampInt32(int64(t))
	default:
		return 0
	}
}

func clampInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

func toInt64(v any) int64 {
	switch t := v.(type) {
	case int:
		return int64(t)
	case int32:
		return int64(t)
	case int64:
		return t
	case float64:
		return int64(t)
	default:
		return 0
	}
}

func toFloat64(v any) float64 {
	switch t := v.(type) {
	case float32:
		return float64(t)
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	default:
		return 0
	}
}

func toBool(v any) bool {
	b, _ := v.(bool)
	return b
}

// asyncEmitter sends decoded readings through the SDK async channel.
type asyncEmitter struct {
	sdk interfaces.DeviceServiceSDK
	ch  chan<- *dsModels.AsyncValues
	lc  logger.LoggingClient
}

func (e *asyncEmitter) Emit(ctx context.Context, device string, readings []decode.Reading, origin int64) {
	if e.ch == nil {
		e.lc.Errorf("Skip async readings for %s: SDK async channel is disabled", device)
		return
	}
	values := make([]*dsModels.CommandValue, 0, len(readings))
	for _, r := range readings {
		if _, ok := e.sdk.DeviceResource(device, r.Name); !ok {
			// A model-specific profile may omit optional or descriptor-named
			// resources; skip them rather than failing the complete event.
			e.lc.Debugf("Skip reading %s for device %s: resource not in profile", r.Name, device)
			continue
		}
		valueType, value := edgeValue(r)
		cv, err := dsModels.NewCommandValueWithOrigin(r.Name, valueType, value, origin)
		if err != nil {
			e.lc.Errorf("skip reading %s for %s: %v", r.Name, device, err)
			continue
		}
		values = append(values, cv)
	}
	if len(values) == 0 {
		return
	}
	event := &dsModels.AsyncValues{
		DeviceName:    device,
		SourceName:    values[0].DeviceResourceName,
		CommandValues: values,
	}
	// Do not block pipeline shutdown indefinitely behind a full async buffer.
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	select {
	case e.ch <- event:
	case <-timer.C:
		e.lc.Errorf("Drop async readings for %s: SDK async buffer is full", device)
	case <-ctx.Done():
		e.lc.Debugf("Drop async readings for %s: service is stopping", device)
	}
}

func edgeValue(r decode.Reading) (string, any) {
	switch r.Type {
	case "int32":
		return common.ValueTypeInt32, toInt32(r.Value)
	case "int64":
		return common.ValueTypeInt64, toInt64(r.Value)
	case "float64":
		return common.ValueTypeFloat64, toFloat64(r.Value)
	case "bool":
		return common.ValueTypeBool, toBool(r.Value)
	case "binary":
		return common.ValueTypeBinary, r.Value
	default:
		return common.ValueTypeString, fmt.Sprintf("%v", r.Value)
	}
}
