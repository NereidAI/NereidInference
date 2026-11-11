// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/modelcatalog"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
	"github.com/edgexfoundry/device-inference/internal/pipeline"
)

// runPipeline is the NI-native multi-model counterpart of run. The legacy
// single-model state machine remains untouched; both share source, scheduling,
// statistics, command invocation and the local EdgeX emitter.
func (j *Job) runPipeline() {
	if j.deps.Pipelines == nil {
		j.stats.ForceState(StateFailed, ReasonConfigInvalid)
		j.deps.Log.Errorf("job %s has PipelineId but no pipeline registry", j.name)
		return
	}
	executor := pipeline.NewExecutor(pipeline.ExecutorDeps{
		Catalog: j.deps.Catalog, Backends: j.deps.Backends, Decoders: j.deps.Decoders,
		Servables: j.deps.Servables,
	})
	resetExecutor := func() {
		executor.Reset()
		j.setPipelineModelRefs(nil)
		j.setPipelineActiveVersion("")
	}
	defer resetExecutor()

	catalogEvents, unsubscribeCatalog := j.deps.Catalog.Subscribe()
	defer unsubscribeCatalog()
	pipelineEvents, unsubscribePipelines := j.deps.Pipelines.Subscribe()
	defer unsubscribePipelines()

	var sub *frame.Subscription
	defer func() {
		if sub != nil {
			sub.Release()
		}
	}()

	resolve := func() bool {
		definition, ok := j.deps.Pipelines.Lookup(j.cfg.PipelineID)
		if !ok {
			j.setPipelineLatestVersion("")
			j.stats.RecordResolutionFailure(ReasonConfigInvalid)
			return false
		}
		// Latest is a registry fact even when its packages/runtime are not ready;
		// active changes only after Executor.Resolve commits the new graph.
		j.setPipelineLatestVersion(definition.Version)
		if err := executor.Resolve(j.ctx, definition, j.cfg.TargetDevice, j.cfg.Endpoint); err != nil {
			reason := ReasonModelNotReady
			var pipelineErr *pipeline.Error
			if errors.As(err, &pipelineErr) && pipelineErr.Reason != "" {
				reason = pipelineErr.Reason
			}
			j.deps.Log.Errorf("job %s pipeline %s resolve: %v", j.name, j.cfg.PipelineID, err)
			j.stats.RecordResolutionFailure(reason)
			return false
		}
		j.setPipelineModelRefs(executor.ModelRefs())
		j.setPipelineActiveVersion(executor.ResolvedVersion())
		j.stats.ClearResolutionFailure()
		return true
	}

	ensureSource := func() error {
		if sub != nil {
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
				case <-catalogEvents:
				case <-pipelineEvents:
				case <-time.After(5 * time.Second):
				}
				continue
			}
			if err := ensureSource(); err != nil {
				recordSourceFailure(err)
			} else {
				j.stats.ForceState(StateApplying, "")
				return true
			}
			select {
			case <-j.ctx.Done():
				return false
			case <-catalogEvents:
			case <-pipelineEvents:
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
	resolveRetry := time.NewTicker(5 * time.Second)
	defer resolveRetry.Stop()

	for {
		select {
		case <-j.ctx.Done():
			return
		case <-pipelineEvents:
			if _, ok := j.deps.Pipelines.Lookup(j.cfg.PipelineID); !ok {
				resetExecutor()
				j.setPipelineLatestVersion("")
				j.stats.RecordResolutionFailure(ReasonConfigInvalid)
				if !waitReady() {
					return
				}
				continue
			}
			// Immutable versions make a same-version notification harmless. A
			// newer accepted version switches only after both models are ready.
			if resolve() {
				j.stats.ForceState(StateApplying, "")
			}
		case ev := <-catalogEvents:
			definition, ok := j.deps.Pipelines.Lookup(j.cfg.PipelineID)
			if !ok {
				continue
			}
			switch pipelineCatalogAction(executor, definition, ev) {
			case pipelineCatalogReset:
				resetExecutor()
				j.stats.RecordResolutionFailure(ReasonModelNotReady)
				if !waitReady() {
					return
				}
			case pipelineCatalogResolve:
				// A failed floating-version probe leaves the prior graph serving.
				if resolve() {
					j.stats.ForceState(StateApplying, "")
				}
			}
		case <-resolveRetry.C:
			definition, ok := j.deps.Pipelines.Lookup(j.cfg.PipelineID)
			if !ok {
				resetExecutor()
				j.setPipelineLatestVersion("")
				j.stats.RecordResolutionFailure(ReasonConfigInvalid)
				continue
			}
			j.setPipelineLatestVersion(definition.Version)
			if executor.ResolvedVersion() == definition.Version && !j.stats.ResolutionFailurePending() {
				continue
			}
			// Server recovery emits no catalog or registry event. Keep using the
			// old graph while retrying a distinct newest version or a failed
			// same-version re-resolution.
			if resolve() {
				j.stats.ForceState(StateApplying, "")
			}
		case req := <-j.invoke:
			if sub == nil {
				if err := ensureSource(); err != nil {
					recordSourceFailure(err)
					req.resp <- invokeResp{err: err}
					continue
				}
			}
			rs, err := j.cyclePipeline(executor, sub)
			if errors.Is(err, errSourceGone) {
				sub = reconnectSource(sub)
			}
			req.resp <- invokeResp{readings: rs, err: err}
		case <-tickerC:
			if sub == nil {
				if err := ensureSource(); err != nil {
					recordSourceFailure(err)
					continue
				}
			}
			_, err := j.cyclePipeline(executor, sub)
			if errors.Is(err, errSourceGone) {
				sub = reconnectSource(sub)
				continue
			}
			if err != nil {
				j.deps.Log.Debugf("job %s pipeline cycle: %v", j.name, err)
			}
		}
	}
}

func pipelineUsesModel(definition pipeline.Definition, id string) bool {
	return definition.Nodes[1].Model.ModelID == id || definition.Nodes[3].Model.ModelID == id
}

type pipelineCatalogActionKind int

const (
	pipelineCatalogIgnore pipelineCatalogActionKind = iota
	pipelineCatalogResolve
	pipelineCatalogReset
)

// pipelineCatalogAction bases destructive removal handling on exact active
// leases. Latest-definition membership alone is insufficient after a failed
// upgrade leaves an older graph serving with a different model set.
func pipelineCatalogAction(executor *pipeline.Executor, definition pipeline.Definition, event modelcatalog.Event) pipelineCatalogActionKind {
	switch event.Kind {
	case modelcatalog.EventRemoved:
		if executor.ReferencesModelVersion(event.ModelID, event.Version) {
			return pipelineCatalogReset
		}
	case modelcatalog.EventAdded:
		if pipelineUsesModel(definition, event.ModelID) || executor.ReferencesModel(event.ModelID) {
			return pipelineCatalogResolve
		}
	}
	return pipelineCatalogIgnore
}

// cyclePipeline performs source acquisition and delegates only the frame-local
// model work to Executor. Raw frames/tensors never enter the emitted reading.
func (j *Job) cyclePipeline(executor *pipeline.Executor, sub *frame.Subscription) (readings []decode.Reading, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			readings = nil
			err = fmt.Errorf("%s: %v", ReasonPostprocessFail, recovered)
			j.stats.RecordError(ReasonPostprocessFail, j.cfg.ErrorThreshold)
		}
	}()
	timeout := 10 * time.Second
	if j.cfg.MaxInferMs > 0 {
		timeout = time.Duration(j.cfg.MaxInferMs) * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(j.ctx, timeout)
	defer cancel()

	j.stats.RecordAttempt()
	start := time.Now()
	source, err := j.nextFrame(ctx, sub)
	if err != nil {
		j.stats.RecordError(reasonFor(err), j.cfg.ErrorThreshold)
		return nil, err
	}
	result, err := executor.Execute(ctx, source, float32(j.cfg.ConfThreshold))
	if err != nil {
		j.stats.RecordError(pipelineFailureReason(err), j.cfg.ErrorThreshold)
		return nil, err
	}
	latency := float64(time.Since(start).Microseconds()) / 1000.0
	readings, err = result.Readings(j.deps.JSONCap)
	if err != nil {
		j.stats.RecordError(ReasonPostprocessFail, j.cfg.ErrorThreshold)
		return nil, err
	}
	readings = append(readings, decode.Reading{Name: "InferLatencyMs", Type: "float64", Value: latency})
	j.stats.RecordSuccess(latency)
	j.cacheReadings(readings)
	if j.deps.Emitter != nil {
		if published := j.filterForPublish(readings); len(published) > 0 {
			j.deps.Emitter.Emit(j.ctx, j.name, published, time.Now().UnixNano())
		}
	}
	return readings, nil
}

func pipelineFailureReason(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return ReasonInferTimeout
	}
	var pipelineErr *pipeline.Error
	if errors.As(err, &pipelineErr) && pipelineErr.Reason != "" {
		return pipelineErr.Reason
	}
	return ReasonPostprocessFail
}

// PipelineModelRefs returns the exact resolved packages for manager binding
// counts. A failed upgrade leaves the prior graph and its references intact.
func (j *Job) PipelineModelRefs() []modelserver.ModelRef {
	j.pipelineModelsMu.RLock()
	defer j.pipelineModelsMu.RUnlock()
	return append([]modelserver.ModelRef(nil), j.pipelineModels...)
}

func (j *Job) setPipelineModelRefs(refs []modelserver.ModelRef) {
	j.pipelineModelsMu.Lock()
	j.pipelineModels = append(j.pipelineModels[:0], refs...)
	j.pipelineModelsMu.Unlock()
}

// PipelineVersions returns the single machine-readable version view for a
// pipeline Job. Active comes only from Executor.ResolvedVersion; latest comes
// only from Registry.Lookup. Empty strings mean no resolved/registered version.
func (j *Job) PipelineVersions() (active, latest string) {
	j.pipelineModelsMu.RLock()
	defer j.pipelineModelsMu.RUnlock()
	return j.pipelineActiveVersion, j.pipelineLatestVersion
}

func (j *Job) setPipelineActiveVersion(version string) {
	j.pipelineModelsMu.Lock()
	j.pipelineActiveVersion = version
	j.pipelineModelsMu.Unlock()
}

func (j *Job) setPipelineLatestVersion(version string) {
	j.pipelineModelsMu.Lock()
	j.pipelineLatestVersion = version
	j.pipelineModelsMu.Unlock()
}
