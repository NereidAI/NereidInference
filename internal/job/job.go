// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/infer"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/pipeline"
	"github.com/edgexfoundry/device-inference/internal/preprocess"
	"github.com/edgexfoundry/device-inference/internal/publish"
)

// Emitter pushes readings into EdgeX as async values.
type Emitter interface {
	Emit(ctx context.Context, device string, rs []decode.Reading, origin int64)
}

// FederationResolver resolves ONVIF-federated camera names to playable
// stream references. *frame.Federation is the production implementation.
type FederationResolver interface {
	Resolve(ctx context.Context, deviceName, secretHint, profileToken string) (frame.ResolvedRef, error)
	Invalidate(deviceName string)
}

// Deps bundles the shared services every pipeline consumes.
type Deps struct {
	Catalog    *modelcatalog.Catalog
	Frames     *frame.Manager
	Federation FederationResolver // optional; nil disables ONVIF federation
	GetSecret  frame.SecretGetter
	Backends   *infer.Registry
	Decoders   *decode.Registry
	Servables  *modelserver.Registry
	Pipelines  *pipeline.Registry
	Preview    *publish.Hub
	Log        Logger
	Emitter    Emitter

	DefaultKServeEndpoint string
	DefaultOpenAIEndpoint string
	CoreCommandURL        string
	JSONCap               int
	FrameTTLFactor        float64
}

// Logger is the small logging surface jobs need.
type Logger interface {
	Infof(format string, args ...any)
	Errorf(format string, args ...any)
	Debugf(format string, args ...any)
}

// Job is one running pipeline bound to one EdgeX device.
type Job struct {
	name string
	cfg  Config
	deps Deps

	stats  *Stats
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	invoke chan invokeReq

	lastMu   sync.RWMutex
	lastMeta preprocess.Meta
	lastVals map[string]any

	pipelineModelsMu      sync.RWMutex
	pipelineModels        []modelserver.ModelRef
	pipelineActiveVersion string
	pipelineLatestVersion string
}

type invokeReq struct {
	prompt string
	resp   chan invokeResp
}

type invokeResp struct {
	readings []decode.Reading
	err      error
}

// New constructs a job; Start launches its loop.
func New(name string, cfg Config, deps Deps) *Job {
	ctx, cancel := context.WithCancel(context.Background())
	return &Job{
		name: name,
		cfg:  cfg,
		deps: deps,
		// A freshly admitted job starts APPLIED while it resolves its model
		// and source. A missing/unready model or invalid binding force FAILED
		// on the first loop; source establishment failures (e.g. a camera
		// secret delivered seconds after the job) escalate through the
		// DEGRADED -> FAILED threshold ladder instead of failing instantly.
		stats:    &Stats{State: StateApplying},
		ctx:      ctx,
		cancel:   cancel,
		invoke:   make(chan invokeReq, 4),
		lastVals: map[string]any{},
	}
}

// Start launches the pipeline.
func (j *Job) Start() {
	j.wg.Add(1)
	go j.run()
}

// Stop performs graceful teardown of the pipeline.
func (j *Job) Stop() {
	j.cancel()
	j.wg.Wait()
}

// Invoke performs one on-demand cycle (event/command triggers and LLM jobs).
func (j *Job) Invoke(ctx context.Context, prompt string) ([]decode.Reading, error) {
	req := invokeReq{prompt: prompt, resp: make(chan invokeResp, 1)}
	select {
	case j.invoke <- req:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case r := <-req.resp:
		return r.readings, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// StatsSnapshot exposes scalar state resources.
func (j *Job) StatsSnapshot() map[string]any { return j.stats.Snapshot() }

// LastValue returns the most recent reading value for a resource.
func (j *Job) LastValue(name string) (any, bool) {
	j.lastMu.RLock()
	defer j.lastMu.RUnlock()
	v, ok := j.lastVals[name]
	return v, ok
}

func (j *Job) run() {
	defer j.wg.Done()
	if j.cfg.PipelineID != "" {
		j.runPipeline()
		return
	}

	events, unsub := j.deps.Catalog.Subscribe()
	defer unsub()

	var (
		entry      *modelcatalog.Entry
		dec        decode.Decoder
		sub        *frame.Subscription
		backend    infer.Backend
		lease      modelserver.Lease
		servedName string
	)
	releaseLease := func() {
		if lease != nil {
			lease.Release()
			lease = nil
		}
		servedName = ""
	}
	defer releaseLease()
	defer func() {
		if sub != nil {
			sub.Release()
		}
	}()

	resolve := func() bool {
		e, ok := j.deps.Catalog.Lookup(j.cfg.ModelID, j.cfg.ModelVersion)
		if !ok {
			return false
		}
		d, err := j.deps.Decoders.Build(e.Manifest)
		if err != nil {
			j.deps.Log.Errorf("job %s decoder build: %v", j.name, err)
			j.stats.ForceState(StateFailed, ReasonConfigInvalid)
			return false
		}
		modelName := e.Manifest.Runtime.ModelName
		endpoint := j.endpointFor(e.Manifest)
		family := e.Manifest.Runtime.Server.Family
		var mgr modelserver.ServableManager
		if j.deps.Servables != nil {
			mgr, _ = j.deps.Servables.For(family)
		}
		if modelserver.IsManagedFamily(family) {
			if mgr == nil {
				j.deps.Log.Errorf("job %s requires a %s model-server manager", j.name, family)
				j.stats.ForceState(StateFailed, ReasonConfigInvalid)
				return false
			}
			if lease != nil {
				// Drop leases for a different package or one invalidated by a
				// package removal/re-distribution while the job kept running;
				// otherwise readiness would be polled forever against a served
				// name the model server no longer configures.
				ref := lease.Ref()
				if ref.ID != e.ID || ref.Version != e.Version || !lease.Current() {
					releaseLease()
				}
			}
			if lease == nil {
				acquired, acquireErr := mgr.Acquire(j.ctx, e, j.cfg.TargetDevice)
				if acquireErr != nil {
					j.deps.Log.Errorf("job %s configure %s servable: %v", j.name, family, acquireErr)
					j.stats.ForceState(StateFailed, ReasonConfigInvalid)
					return false
				}
				if acquired == nil {
					j.deps.Log.Errorf("job %s configure %s servable: manager returned a nil lease", j.name, family)
					j.stats.ForceState(StateFailed, ReasonConfigInvalid)
					return false
				}
				lease = acquired
			}
			modelName = lease.ModelName()
			endpoint = lease.Endpoint()
			if endpoint == "" {
				j.deps.Log.Errorf("job %s configure %s servable: lease has no backend endpoint", j.name, family)
				j.stats.ForceState(StateFailed, ReasonConfigInvalid)
				return false
			}
			if j.cfg.Endpoint != "" {
				// Keep the original explicit override for existing single-model
				// deployments; the managed lease still gates name and readiness.
				endpoint = j.cfg.Endpoint
			}
		}
		b, err := j.deps.Backends.Get(e.Manifest.Runtime.Backend, endpoint)
		if err != nil {
			j.deps.Log.Errorf("job %s backend: %v", j.name, err)
			j.stats.ForceState(StateFailed, ReasonConfigInvalid)
			return false
		}
		if err := b.Ready(j.ctx, modelName, ""); err != nil {
			j.deps.Log.Errorf("job %s model %s/%s is not ready: %v", j.name, j.cfg.ModelID, j.cfg.ModelVersion, err)
			j.stats.ForceState(StateFailed, ReasonModelNotReady)
			return false
		}
		entry, dec, backend, servedName = e, d, b, modelName
		return true
	}

	ensureSource := func() error {
		if sub != nil || j.cfg.SourceKind == "none" {
			return nil
		}
		ref, err := j.resolveRef()
		if err != nil {
			return err
		}
		s, err := j.deps.Frames.Subscribe(j.ctx, ref)
		if err != nil {
			return err
		}
		sub = s
		return nil
	}
	reconnectSource := func(old *frame.Subscription) *frame.Subscription {
		if j.cfg.SourceKind == "onvif-camera" && j.deps.Federation != nil {
			// A setup/auth failure may mean the cached GetStreamUri result or
			// its short-lived token rotated; force a fresh federation resolve.
			j.deps.Federation.Invalidate(j.cfg.SourceDevice)
		}
		if old != nil {
			old.Release()
		}
		sub = nil
		if err := ensureSource(); err != nil {
			j.deps.Log.Errorf("job %s source reconnect failed: %v", j.name, err)
			return nil
		}
		return sub
	}
	// recordSourceFailure accounts for a source establishment failure so a
	// permanently unreachable stream (federation resolve, secret injection or
	// ffmpeg startup) escalates through the same error thresholds as frame-less
	// cycles: DEGRADED then FAILED/source-no-frame, instead of being invisible
	// and later relabelled idle-no-inference by the watchdog. The underlying
	// error text stays in service logs; LastError only carries the constant.
	recordSourceFailure := func(err error) {
		j.stats.RecordAttempt()
		j.stats.RecordError(ReasonSourceNoFrame, j.cfg.ErrorThreshold)
		j.deps.Log.Debugf("job %s source unavailable: %v", j.name, err)
	}

	waitReady := func() bool {
		for {
			if !resolve() {
				select {
				case <-j.ctx.Done():
					return false
				case <-events:
				case <-time.After(5 * time.Second):
				}
				continue
			}
			if err := ensureSource(); err != nil {
				// Startup races (e.g. NE delivers the job seconds before the
				// camera secret) are transient: account them through the same
				// DEGRADED -> FAILED threshold ladder as frame-less cycles and
				// keep retrying, instead of force-failing on the first miss.
				recordSourceFailure(err)
			} else {
				j.stats.ForceState(StateApplying, "")
				return true
			}
			select {
			case <-j.ctx.Done():
				return false
			case <-events:
			case <-time.After(5 * time.Second):
			}
		}
	}
	if !waitReady() {
		return
	}
	j.wg.Add(1)
	go j.watchdog()

	var tickerC <-chan time.Time
	var ticker *time.Ticker
	switch j.cfg.Trigger {
	case TriggerContinuous:
		interval := time.Second / time.Duration(max1(j.cfg.FPS))
		if j.cfg.FPS <= 0 {
			interval = time.Second
		}
		ticker = time.NewTicker(interval)
		tickerC = ticker.C
	case TriggerPeriodic:
		d := time.Duration(j.cfg.PeriodMs) * time.Millisecond
		if d <= 0 {
			d = time.Second
		}
		ticker = time.NewTicker(d)
		tickerC = ticker.C
	}
	if ticker != nil {
		defer ticker.Stop()
	}

	for {
		select {
		case <-j.ctx.Done():
			return
		case ev := <-events:
			if ev.ModelID != j.cfg.ModelID {
				continue
			}
			switch ev.Kind {
			case modelcatalog.EventRemoved:
				if entry != nil && j.cfg.ModelVersion != "" &&
					ev.Version != entry.Version {
					continue
				}
				if current, ok := j.deps.Catalog.Lookup(j.cfg.ModelID, j.cfg.ModelVersion); ok &&
					entry != nil && current.Version == entry.Version {
					continue
				}
				entry = nil
				releaseLease()
				if sub != nil {
					sub.Release()
					sub = nil
				}
				j.stats.ForceState(StateFailed, ReasonModelNotReady)
				if !waitReady() {
					return
				}
			case modelcatalog.EventAdded:
				if j.cfg.ModelVersion != "" && ev.Version != j.cfg.ModelVersion {
					continue
				}
				// Unpinned jobs may float to the newest installed package.
				// A failed probe keeps serving the prior entry until the next
				// catalog event rather than tearing down a working pipeline.
				if resolve() {
					j.stats.ForceState(StateApplying, "")
				}
			}
		case req := <-j.invoke:
			if sub == nil && j.cfg.SourceKind != "none" {
				if err := ensureSource(); err != nil {
					recordSourceFailure(err)
					req.resp <- invokeResp{err: err}
					continue
				}
			}
			rs, err := j.cycle(backend, dec, entry, servedName, sub, req.prompt)
			if errors.Is(err, errSourceGone) {
				sub = reconnectSource(sub)
			}
			req.resp <- invokeResp{readings: rs, err: err}
		case <-tickerC:
			if sub == nil && j.cfg.SourceKind != "none" {
				if err := ensureSource(); err != nil {
					recordSourceFailure(err)
					continue
				}
			}
			_, err := j.cycle(backend, dec, entry, servedName, sub, "")
			if errors.Is(err, errSourceGone) {
				sub = reconnectSource(sub)
				continue
			}
			if err != nil {
				j.deps.Log.Debugf("job %s cycle: %v", j.name, err)
			}
		}
	}
}

func (j *Job) endpointFor(m *model.Manifest) string {
	if j.cfg.Endpoint != "" {
		return j.cfg.Endpoint
	}
	if m.Runtime.Backend == "openai" {
		return j.deps.DefaultOpenAIEndpoint
	}
	return j.deps.DefaultKServeEndpoint
}

func (j *Job) resolveRef() (frame.ResolvedRef, error) {
	if j.cfg.SourceKind == "onvif-camera" {
		if j.deps.Federation == nil {
			return frame.ResolvedRef{}, fmt.Errorf("onvif federation is not configured")
		}
		ref, err := j.deps.Federation.Resolve(j.ctx, j.cfg.SourceDevice, j.cfg.SecretName, j.cfg.StreamProfile)
		if err != nil {
			return frame.ResolvedRef{}, err
		}
		// Per-job ffmpeg tuning applies to the federated stream exactly as
		// it does for direct rtsp sources (allowlist-validated in config).
		ref.ExtraArgs = j.cfg.FFmpegExtra
		ref.DecodeAccel = j.cfg.DecodeAccel
		ref.VAAPIDevice = j.cfg.VAAPIDevice
		return ref, nil
	}
	if j.cfg.SourceKind == "onvif-snapshot" {
		if j.deps.CoreCommandURL == "" {
			return frame.ResolvedRef{}, fmt.Errorf("core-command URL is not configured")
		}
		snapshotURL := strings.TrimRight(j.deps.CoreCommandURL, "/") +
			"/api/v3/device/name/" + url.PathEscape(j.cfg.SourceDevice) + "/Snapshot"
		return frame.ResolvedRef{Kind: "onvif-snapshot", URL: snapshotURL}, nil
	}
	kind := j.cfg.SourceKind
	if kind == "" {
		kind = "rtsp"
	}
	rawURL := j.cfg.SourceURL
	if j.cfg.SecretName != "" && j.deps.GetSecret != nil {
		injected, err := frame.InjectCredentials(rawURL, j.cfg.SecretName, j.deps.GetSecret)
		if err != nil {
			return frame.ResolvedRef{}, err
		}
		rawURL = injected
	}
	return frame.ResolvedRef{
		Kind:        kind,
		URL:         rawURL,
		ExtraArgs:   j.cfg.FFmpegExtra,
		DecodeAccel: j.cfg.DecodeAccel,
		VAAPIDevice: j.cfg.VAAPIDevice,
	}, nil
}

func (j *Job) watchdog() {
	defer j.wg.Done()
	if j.cfg.Trigger != TriggerContinuous && j.cfg.Trigger != TriggerPeriodic {
		return
	}
	window := time.Duration(j.cfg.MaxIdleMs) * time.Millisecond
	if window <= 0 {
		window = 3 * j.scheduleInterval()
		if window < 3*time.Second {
			window = 3 * time.Second
		}
	}
	t := time.NewTicker(10 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-j.ctx.Done():
			return
		case <-t.C:
			j.stats.MarkIdle(window)
		}
	}
}

// cycle executes one full acquire -> build -> infer -> decode cycle.
func (j *Job) cycle(backend infer.Backend, dec decode.Decoder, entry *modelcatalog.Entry, modelName string, sub *frame.Subscription, promptArg string) (readings []decode.Reading, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			readings = nil
			err = fmt.Errorf("%s: %v", ReasonPostprocessFail, recovered)
			j.stats.RecordError(ReasonPostprocessFail, j.cfg.ErrorThreshold)
		}
	}()
	if entry == nil {
		return nil, fmt.Errorf("model is not ready")
	}
	m := entry.Manifest
	timeout := 10 * time.Second
	if j.cfg.MaxInferMs > 0 {
		timeout = time.Duration(j.cfg.MaxInferMs) * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(j.ctx, timeout)
	defer cancel()

	j.stats.RecordAttempt()
	start := time.Now()

	req, fr, err := j.buildRequest(ctx, m, modelName, sub, promptArg)
	if err != nil {
		j.stats.RecordError(reasonFor(err), j.cfg.ErrorThreshold)
		return nil, err
	}
	resp, err := backend.Infer(ctx, req)
	if err != nil {
		j.stats.RecordError(ReasonRuntimeError, j.cfg.ErrorThreshold)
		return nil, err
	}
	result, err := dec.Decode(&decode.Input{
		Manifest:     m,
		Labels:       entry.Labels,
		Response:     resp,
		Meta:         j.currentMeta(),
		ConfOverride: float32(j.cfg.ConfThreshold),
	})
	if err != nil {
		j.stats.RecordError(ReasonPostprocessFail, j.cfg.ErrorThreshold)
		return nil, err
	}
	latency := float64(time.Since(start).Microseconds()) / 1000.0
	readings, err = result.Readings(j.deps.JSONCap)
	if err != nil {
		j.stats.RecordError(ReasonPostprocessFail, j.cfg.ErrorThreshold)
		return nil, err
	}
	readings = append(readings, decode.Reading{Name: "InferLatencyMs", Type: "float64", Value: latency})
	var previewFrame *publish.Frame
	if fr != nil {
		previewFrame = &publish.Frame{
			Source: fr, Result: result, Labels: entry.Labels,
			Annotate: j.cfg.AnnotatePreview,
		}
	}
	if j.cfg.EmitFrame && previewFrame != nil {
		if b, err := publish.EncodeJPEG(previewFrame, j.cfg.AnnotatePreview); err == nil {
			readings = append(readings, decode.Reading{Name: "FrameJPEG", Type: "binary", Value: b})
		}
	}
	j.stats.RecordSuccess(latency)

	j.cacheReadings(readings)
	if j.deps.Emitter != nil {
		if published := j.filterForPublish(readings); len(published) > 0 {
			j.deps.Emitter.Emit(j.ctx, j.name, published, time.Now().UnixNano())
		}
	}
	if j.deps.Preview != nil && previewFrame != nil {
		j.deps.Preview.Update(j.name, previewFrame)
	}
	return readings, nil
}

// buildRequest constructs the backend request selected by the descriptor.
func (j *Job) buildRequest(ctx context.Context, m *model.Manifest, modelName string, sub *frame.Subscription, promptArg string) (infer.Request, *frame.Frame, error) {
	// The package version identifies the immutable active tree; the model
	// server's numeric model version is selected by the server when this
	// field is empty (KServe v2 standard behavior).
	req := infer.Request{ModelName: modelName, SamplingJSON: j.cfg.SamplingJSON}
	if m.Runtime.Backend == "openai" {
		return j.buildOpenAIRequest(ctx, m, sub, promptArg, req)
	}

	if len(m.IO.Inputs) == 0 {
		return req, nil, fmt.Errorf("manifest declares no inputs")
	}
	fr, err := j.nextFrame(ctx, sub)
	if err != nil {
		return req, nil, err
	}
	tensor, err := preprocess.Build(fr, m.IO.Inputs[0])
	if err != nil {
		return req, fr, err
	}
	j.setMeta(tensor.Meta)
	req.Inputs = []infer.TensorInput{{
		Name:     m.IO.Inputs[0].Name,
		Shape:    tensor.Shape,
		Float32:  tensor.Data,
		DataType: "FP32",
	}}
	return req, fr, nil
}

func (j *Job) buildOpenAIRequest(ctx context.Context, m *model.Manifest, sub *frame.Subscription, promptArg string, req infer.Request) (infer.Request, *frame.Frame, error) {
	if strings.EqualFold(m.Model.Task, "embeddings") {
		text := strings.TrimSpace(firstNonEmpty(promptArg, j.cfg.PromptTemplate))
		if text == "" {
			return req, nil, fmt.Errorf("embeddings request requires prompt text")
		}
		req.Embeddings = []string{text}
		return req, nil, nil
	}

	prompt := firstNonEmpty(promptArg, j.cfg.PromptTemplate)
	if prompt == "" {
		prompt = "Describe this image concisely."
	}
	if j.cfg.SourceKind == "none" || sub == nil {
		req.Messages = []infer.Message{{Role: "user", Content: prompt}}
		return req, nil, nil
	}
	fr, err := j.nextFrame(ctx, sub)
	if err != nil {
		return req, nil, err
	}
	b, err := jpegBytes(fr)
	if err != nil {
		return req, fr, err
	}
	dataURI := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(b)
	req.Messages = []infer.Message{{
		Role: "user",
		Content: []map[string]any{
			{"type": "text", "text": prompt},
			{"type": "image_url", "image_url": map[string]string{"url": dataURI}},
		},
	}}
	return req, fr, nil
}

// nextFrame returns the newest frame whose age is within the configured TTL.
// The fan-out subscription is buffer 1 with drop-oldest semantics; this loop
// additionally protects a slow inference cycle from processing stale media.
func (j *Job) nextFrame(ctx context.Context, sub *frame.Subscription) (*frame.Frame, error) {
	if sub == nil {
		return nil, fmt.Errorf("frame source is not available")
	}
	ttl := j.frameTTL()
	wait := 2 * ttl
	if wait < 2*time.Second {
		wait = 2 * time.Second
	}
	deadline := time.Now().Add(wait)
	var latest *frame.Frame
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			if latest != nil {
				return nil, fmt.Errorf("%w: newest frame is %s old, ttl %s", errNoFrame, time.Since(latest.Time), ttl)
			}
			return nil, fmt.Errorf("%w: no frame within %s", errNoFrame, wait)
		}
		timer := time.NewTimer(remaining)
		select {
		case fr, ok := <-sub.C:
			timer.Stop()
			if !ok {
				return nil, errSourceGone
			}
			if time.Since(fr.Time) <= ttl {
				return &fr, nil
			}
			candidate := fr
			latest = &candidate
		case <-timer.C:
			if latest != nil {
				return nil, fmt.Errorf("%w: newest frame is %s old, ttl %s", errNoFrame, time.Since(latest.Time), ttl)
			}
			return nil, fmt.Errorf("%w: no frame within %s", errNoFrame, wait)
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		}
	}
}

func (j *Job) scheduleInterval() time.Duration {
	switch j.cfg.Trigger {
	case TriggerPeriodic:
		if j.cfg.PeriodMs > 0 {
			return time.Duration(j.cfg.PeriodMs) * time.Millisecond
		}
		return time.Second
	case TriggerContinuous:
		return time.Second / time.Duration(max1(j.cfg.FPS))
	default:
		return time.Second
	}
}

func (j *Job) frameTTL() time.Duration {
	factor := j.deps.FrameTTLFactor
	if factor <= 0 {
		factor = 2
	}
	ttl := time.Duration(float64(j.scheduleInterval()) * factor)
	if ttl <= 0 {
		return 2 * time.Second
	}
	return ttl
}

var (
	errNoFrame    = fmt.Errorf("source-no-frame")
	errSourceGone = fmt.Errorf("source-closed")
)

func reasonFor(err error) string {
	switch {
	case err == nil:
		return ""
	case err == context.DeadlineExceeded:
		return ReasonInferTimeout
	case errors.Is(err, errSourceGone):
		return ReasonSourceNoFrame
	case strings.Contains(err.Error(), "no frame"):
		return ReasonSourceNoFrame
	default:
		return ReasonPreprocessFail
	}
}

func (j *Job) currentMeta() preprocess.Meta {
	j.lastMu.RLock()
	defer j.lastMu.RUnlock()
	return j.lastMeta
}

func (j *Job) setMeta(meta preprocess.Meta) {
	j.lastMu.Lock()
	j.lastMeta = meta
	j.lastMu.Unlock()
}

func (j *Job) cacheReadings(rs []decode.Reading) {
	j.lastMu.Lock()
	defer j.lastMu.Unlock()
	for k, v := range j.stats.Snapshot() {
		j.lastVals[k] = v
	}
	for _, r := range rs {
		j.lastVals[r.Name] = r.Value
	}
}

// jpegBytes encodes an acquired frame for VLM or binary frame readings.
func jpegBytes(f *frame.Frame) ([]byte, error) {
	if f.Encoding == frame.JPEG {
		return append([]byte(nil), f.Bytes...), nil
	}
	if f.Encoding != frame.RGB24 || len(f.Bytes) < f.Width*f.Height*3 {
		return nil, fmt.Errorf("invalid rgb24 frame")
	}
	img := image.NewRGBA(image.Rect(0, 0, f.Width, f.Height))
	for i := 0; i < f.Width*f.Height; i++ {
		img.Pix[4*i] = f.Bytes[3*i]
		img.Pix[4*i+1] = f.Bytes[3*i+1]
		img.Pix[4*i+2] = f.Bytes[3*i+2]
		img.Pix[4*i+3] = 255
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func max1(v float64) float64 {
	if v < 1 {
		return 1
	}
	return v
}
