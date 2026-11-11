// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"sort"
	"sync"
	"time"
)

// State values mirror the transport-safe connector status constants.
const (
	StateApplying = "APPLIED"
	StateDegraded = "DEGRADED"
	StateFailed   = "FAILED"
)

// Transport-safe failure reasons (platform contract section 8.3).
const (
	ReasonModelNotReady   = "model-not-ready"
	ReasonRuntimeUnreach  = "runtime-unreachable"
	ReasonRuntimeError    = "runtime-error"
	ReasonInferTimeout    = "inference-timeout"
	ReasonSourceNoFrame   = "source-no-frame"
	ReasonPreprocessFail  = "preprocess-failed"
	ReasonPostprocessFail = "postprocess-failed"
	ReasonIdleNoInference = "idle-no-inference"
	ReasonResourceExhaust = "resource-exhausted"
	ReasonConfigInvalid   = "config-invalid"
)

// Stats is the per-job statistics snapshot.
type Stats struct {
	mu sync.Mutex

	State  string
	Reason string

	lastSuccess  time.Time
	lastAttempt  time.Time
	successTotal int64
	errorTotal   int64
	consecutive  int
	latencies    []float64

	// resolutionFailure keeps a failed newest pipeline resolution visible while
	// an older graph continues to serve successfully. Only a successful resolve
	// of the currently latest definition may clear this latch.
	resolutionFailure bool
}

// RecordAttempt marks one inference attempt.
func (s *Stats) RecordAttempt() {
	s.mu.Lock()
	s.lastAttempt = time.Now()
	s.mu.Unlock()
}

// RecordSuccess records a completed cycle with latency milliseconds.
func (s *Stats) RecordSuccess(latencyMs float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastSuccess = time.Now()
	s.successTotal++
	s.consecutive = 0
	s.latencies = append(s.latencies, latencyMs)
	if len(s.latencies) > 256 {
		s.latencies = s.latencies[len(s.latencies)-256:]
	}
	// Serving an older graph is still a real success, but it must not make a
	// failed newest-version resolution look healthy.
	if s.resolutionFailure {
		return
	}
	// An ordinary success clears the last failure reason: a sub-threshold
	// failure leaves the job APPLIED but stains LastError, and without this
	// clear the reason would linger long after the streak healed.
	if s.State != StateApplying {
		s.State = StateApplying
	}
	s.Reason = ""
}

// failedEscalationFactor converts the degrade threshold into the hard-fail
// threshold: failures persisting this many times longer leave no doubt that
// the job cannot serve, so it reports FAILED instead of DEGRADED. A later
// successful cycle resets the streak and returns the job to APPLIED, so
// FAILED stays self-healing.
const failedEscalationFactor = 6

// RecordError counts a failure. At the configured threshold the job becomes
// DEGRADED; once errors persist for failedEscalationFactor thresholds it
// becomes FAILED. The last reason always reflects the most recent failure.
func (s *Stats) RecordError(reason string, threshold int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.errorTotal++
	s.consecutive++
	if s.resolutionFailure {
		return
	}
	s.Reason = reason
	if threshold <= 0 {
		threshold = 5
	}
	hard := threshold * failedEscalationFactor
	if hard < threshold { // integer overflow guard
		hard = threshold
	}
	// State always follows the consecutive-failure count: below the
	// threshold the job stays nominal, so a prior FAILED/model-not-ready
	// startup state cannot freeze a job on its first transient source
	// failure.
	switch {
	case s.consecutive >= hard:
		s.State = StateFailed
	case s.consecutive >= threshold:
		s.State = StateDegraded
	default:
		s.State = StateApplying
	}
}

// ForceState sets the state directly (model catalog events). A pending
// latest-version resolution failure takes precedence over old-graph activity.
func (s *Stats) ForceState(state, reason string) {
	s.mu.Lock()
	if !s.resolutionFailure {
		s.State, s.Reason = state, reason
	}
	s.mu.Unlock()
}

// RecordResolutionFailure latches a newest-definition resolution failure.
// Repeated retries may refine the reason, but old-graph successes cannot clear
// it. ErrorTotal remains an inference-attempt counter, not a poll counter.
func (s *Stats) RecordResolutionFailure(reason string) {
	s.mu.Lock()
	s.resolutionFailure = true
	s.State, s.Reason = StateFailed, reason
	s.mu.Unlock()
}

// ClearResolutionFailure acknowledges successful resolution of the latest
// definition. It does not clear unrelated inference health when no resolution
// failure is pending.
func (s *Stats) ClearResolutionFailure() {
	s.mu.Lock()
	if s.resolutionFailure {
		s.resolutionFailure = false
		s.State, s.Reason = StateApplying, ""
	}
	s.mu.Unlock()
}

// ResolutionFailurePending reports whether the retry loop must re-resolve even
// when active and latest versions are already equal.
func (s *Stats) ResolutionFailurePending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.resolutionFailure
}

// Snapshot returns the scalar resources for read commands.
func (s *Stats) Snapshot() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string]any{
		"State":        s.State,
		"LastError":    s.Reason,
		"SuccessTotal": int64Value(s.successTotal),
		"ErrorTotal":   int64Value(s.errorTotal),
		"P50Ms":        percentile(s.latencies, 0.5),
		"P95Ms":        percentile(s.latencies, 0.95),
	}
	if !s.lastSuccess.IsZero() {
		out["LastSuccessAt"] = s.lastSuccess.UnixMilli()
	}
	if !s.lastAttempt.IsZero() {
		// Idle watchdog: continuous/periodic jobs expect frames on schedule.
		out["LastAttemptAt"] = s.lastAttempt.UnixMilli()
	}
	return out
}

// MarkIdle flags a healthy job that stopped producing attempts or successes
// within the window as DEGRADED/idle-no-inference. It deliberately never
// overrides a non-APPLYING state: a job with recorded failures already carries
// the precise error reason (e.g. source-no-frame) and may be on its way to
// FAILED; the idle watchdog must not relabel that as idle-no-inference.
func (s *Stats) MarkIdle(window time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.State != StateApplying {
		return
	}
	var idle bool
	if s.lastSuccess.IsZero() {
		idle = !s.lastAttempt.IsZero() && time.Since(s.lastAttempt) > window
	} else {
		idle = time.Since(s.lastSuccess) > window
	}
	if idle {
		s.State = StateDegraded
		s.Reason = ReasonIdleNoInference
	}
}

func int64Value(v int64) int64 { return v }

func percentile(samples []float64, q float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	cp := append([]float64(nil), samples...)
	sort.Float64s(cp)
	idx := int(float64(len(cp)-1) * q)
	return cp[idx]
}
