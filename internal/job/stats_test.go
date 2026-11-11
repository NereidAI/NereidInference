// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"testing"
	"time"
)

func TestResolutionFailureSurvivesOldGraphActivityUntilResolved(t *testing.T) {
	s := &Stats{State: StateApplying}
	s.RecordSuccess(1.5)
	s.RecordResolutionFailure(ReasonModelNotReady)
	for attempt := 0; attempt < 3; attempt++ {
		s.RecordSuccess(2.5)
		s.RecordAttempt()
		s.RecordError(ReasonRuntimeUnreach, 1)
		s.ForceState(StateApplying, "")
		got := s.Snapshot()
		if got["State"] != StateFailed || got["LastError"] != ReasonModelNotReady {
			t.Fatalf("old-graph activity %d cleared latest resolution failure: %+v", attempt, got)
		}
	}
	got := s.Snapshot()
	if got["SuccessTotal"] != int64(4) || got["ErrorTotal"] != int64(3) {
		t.Fatalf("resolution latch discarded attempt statistics: %+v", got)
	}
	s.ClearResolutionFailure()
	got = s.Snapshot()
	if got["State"] != StateApplying || got["LastError"] != "" {
		t.Fatalf("successful latest resolution did not clear the latch: %+v", got)
	}
}

func TestRecordErrorTwoTierEscalation(t *testing.T) {
	s := &Stats{State: StateApplying}
	const threshold = 2

	// Below the threshold nothing changes.
	s.RecordError(ReasonSourceNoFrame, threshold)
	if s.State != StateApplying || s.Reason != ReasonSourceNoFrame {
		t.Fatalf("after 1 error: state=%q reason=%q", s.State, s.Reason)
	}

	// At the threshold: DEGRADED.
	s.RecordError(ReasonSourceNoFrame, threshold)
	if s.State != StateDegraded {
		t.Fatalf("after %d errors: state=%q, want DEGRADED", threshold, s.State)
	}

	// Persistent errors (6x threshold): FAILED with the same precise reason.
	for i := 0; i < threshold*failedEscalationFactor-2; i++ {
		s.RecordError(ReasonSourceNoFrame, threshold)
	}
	if s.State != StateFailed || s.Reason != ReasonSourceNoFrame {
		t.Fatalf("after %d errors: state=%q reason=%q, want FAILED/%s",
			threshold*failedEscalationFactor, s.State, s.Reason, ReasonSourceNoFrame)
	}

	// A later success self-heals the job and clears the reason.
	s.RecordSuccess(1.5)
	if s.State != StateApplying || s.Reason != "" {
		t.Fatalf("after recovery: state=%q reason=%q, want APPLIED/empty", s.State, s.Reason)
	}
}

func TestSubthresholdFailureReasonClearsOnSuccess(t *testing.T) {
	s := &Stats{State: StateApplying}
	s.RecordSuccess(1.0)
	// One isolated failure does not leave APPLIED, but records a reason.
	s.RecordError(ReasonSourceNoFrame, 5)
	if s.State != StateApplying || s.Reason != ReasonSourceNoFrame {
		t.Fatalf("after 1 error: state=%q reason=%q", s.State, s.Reason)
	}
	// The next successful cycle must clear the stale reason even though the
	// state never left APPLIED.
	s.RecordSuccess(1.0)
	if s.State != StateApplying || s.Reason != "" {
		t.Fatalf("after healing success: state=%q reason=%q, want APPLIED/empty", s.State, s.Reason)
	}
}

func TestRecordErrorDefaultThreshold(t *testing.T) {
	s := &Stats{State: StateApplying}
	for i := 0; i < 4; i++ {
		s.RecordError(ReasonRuntimeError, 0)
	}
	if s.State != StateApplying {
		t.Fatalf("state=%q, want APPLIED before default threshold", s.State)
	}
	s.RecordError(ReasonRuntimeError, 0)
	if s.State != StateDegraded {
		t.Fatalf("state=%q, want DEGRADED at default threshold 5", s.State)
	}
}

func TestMarkIdleDoesNotOverrideErrorState(t *testing.T) {
	s := &Stats{State: StateFailed, Reason: ReasonSourceNoFrame}
	s.RecordAttempt()
	s.MarkIdle(time.Millisecond)
	if s.State != StateFailed || s.Reason != ReasonSourceNoFrame {
		t.Fatalf("idle watchdog overrode failure: state=%q reason=%q", s.State, s.Reason)
	}

	// A healthy job with no success for longer than the window does degrade.
	healthy := &Stats{State: StateApplying}
	healthy.RecordAttempt()
	time.Sleep(2 * time.Millisecond)
	healthy.MarkIdle(time.Millisecond)
	if healthy.State != StateDegraded || healthy.Reason != ReasonIdleNoInference {
		t.Fatalf("state=%q reason=%q, want DEGRADED/idle-no-inference", healthy.State, healthy.Reason)
	}

	// A job that never attempted anything stays untouched.
	fresh := &Stats{State: StateApplying}
	fresh.MarkIdle(time.Millisecond)
	if fresh.State != StateApplying || fresh.Reason != "" {
		t.Fatalf("fresh job marked idle: state=%q reason=%q", fresh.State, fresh.Reason)
	}
}
