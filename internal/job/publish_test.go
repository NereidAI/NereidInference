// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"testing"

	"github.com/edgexfoundry/device-inference/internal/decode"
)

func TestPublishModeFromProperties(t *testing.T) {
	cfg, err := FromProperties(map[string]any{
		"ModelID":         "m1",
		"SourceKind":      "none",
		"PublishMode":     "NonEmpty",
		"PublishReadings": "DetectionJSON, Top1Label, DetectionJSON",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.PublishMode != PublishNonEmpty {
		t.Fatalf("PublishMode = %q, want %q", cfg.PublishMode, PublishNonEmpty)
	}
	want := []string{"DetectionJSON", "Top1Label"}
	if len(cfg.PublishReadings) != len(want) {
		t.Fatalf("PublishReadings = %v, want %v", cfg.PublishReadings, want)
	}
	for i := range want {
		if cfg.PublishReadings[i] != want[i] {
			t.Fatalf("PublishReadings = %v, want %v", cfg.PublishReadings, want)
		}
	}

	if _, err := FromProperties(map[string]any{"ModelID": "m1", "SourceKind": "none", "PublishMode": "weekly"}); err == nil {
		t.Fatal("invalid PublishMode must be rejected")
	}
	if _, err := FromProperties(map[string]any{"ModelID": "m1", "SourceKind": "none", "PublishReadings": "bad name"}); err == nil {
		t.Fatal("invalid reading name must be rejected")
	}
}

func TestPublishReadingsCSVOmittedEmptyCaseAndValidation(t *testing.T) {
	base := func() map[string]any { return map[string]any{"ModelID": "m1", "SourceKind": "none"} }
	for _, input := range []string{"", "   ", ", ,  ,"} {
		props := base()
		props["PublishReadings"] = input
		cfg, err := FromProperties(props)
		if err != nil || len(cfg.PublishReadings) != 0 {
			t.Errorf("PublishReadings %q = %v, %v; want an empty unfiltered list", input, cfg.PublishReadings, err)
		}
	}
	cfg, err := FromProperties(base())
	if err != nil || len(cfg.PublishReadings) != 0 {
		t.Fatalf("omitted PublishReadings = %v, %v; want empty", cfg.PublishReadings, err)
	}
	props := base()
	props["PublishReadings"] = " DetectionJSON, ,Top1Label,detectionjson,DetectionJSON "
	cfg, err = FromProperties(props)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"DetectionJSON", "Top1Label", "detectionjson"}
	if len(cfg.PublishReadings) != len(want) {
		t.Fatalf("CSV normalization = %v, want %v", cfg.PublishReadings, want)
	}
	for i := range want {
		if cfg.PublishReadings[i] != want[i] {
			t.Fatalf("CSV normalization = %v, want case-preserving deduplicated %v", cfg.PublishReadings, want)
		}
	}
	j := &Job{cfg: Config{PublishMode: PublishAll, PublishReadings: cfg.PublishReadings}}
	readings := []decode.Reading{{Name: "DetectionJSON", Type: "string", Value: "[]"}}
	if got := j.filterForPublish(readings); len(got) != 1 {
		t.Fatalf("case-preserving whitelist rejected its exact resource name: %v", got)
	}
	j.cfg.PublishReadings = []string{"detectionjson"}
	if got := j.filterForPublish(readings); len(got) != 0 {
		t.Fatalf("resource-name matching must be case-sensitive: %v", got)
	}
	for _, invalid := range []string{"bad name", "1Detection", "Detection-JSON", "DetectionJSON;Drop"} {
		props := base()
		props["PublishReadings"] = invalid
		if _, err := FromProperties(props); err == nil {
			t.Errorf("invalid PublishReadings name %q must be rejected", invalid)
		}
	}
}

func TestFilterForPublishModes(t *testing.T) {
	readings := []decode.Reading{
		{Name: "DetectionCount", Type: "int32", Value: int32(0)},
		{Name: "DetectionJSON", Type: "string", Value: "[]"},
		{Name: "InferLatencyMs", Type: "float64", Value: 1.2},
	}
	j := &Job{cfg: Config{PublishMode: PublishAll}}
	if got := j.filterForPublish(readings); len(got) != 3 {
		t.Fatalf("all mode must keep every reading, got %d", len(got))
	}

	j.cfg.PublishMode = PublishNone
	if got := j.filterForPublish(readings); got != nil {
		t.Fatalf("none mode must suppress emission, got %v", got)
	}

	j.cfg.PublishMode = PublishNonEmpty
	if got := j.filterForPublish(readings); len(got) != 0 {
		t.Fatalf("empty frame must be suppressed in nonempty mode, got %v", got)
	}

	withContent := append([]decode.Reading{}, readings...)
	withContent[1] = decode.Reading{Name: "DetectionJSON", Type: "string", Value: `[{"label":"car"}]`}
	withContent[0] = decode.Reading{Name: "DetectionCount", Type: "int32", Value: int32(1)}
	if got := j.filterForPublish(withContent); len(got) != 3 {
		t.Fatalf("nonempty mode must publish content frames, got %d", len(got))
	}

	// Scalar regressions always count as content.
	regression := []decode.Reading{
		{Name: "Value", Type: "float64", Value: 42.5},
		{Name: "InferLatencyMs", Type: "float64", Value: 1.2},
	}
	if got := j.filterForPublish(regression); len(got) != 2 {
		t.Fatalf("regression frame must count as business content, got %d", len(got))
	}
}

func TestFilterPipelineEmptyObjectsForNonEmptyMode(t *testing.T) {
	j := &Job{cfg: Config{PublishMode: PublishNonEmpty}}
	empty := []decode.Reading{{Name: "PipelineJSON", Type: "string", Value: `{"pipelineId":"face","frameId":1,"objects":[]}`}}
	if got := j.filterForPublish(empty); len(got) != 0 {
		t.Fatalf("empty pipeline frame must be suppressed: %v", got)
	}
	nonempty := []decode.Reading{{Name: "PipelineJSON", Type: "string", Value: `{"pipelineId":"face","frameId":2,"objects":[{"id":1}]}`}}
	if got := j.filterForPublish(nonempty); len(got) != 1 {
		t.Fatalf("nonempty pipeline frame must be published: %v", got)
	}
}

func TestFilterForPublishModeRunsBeforeWhitelist(t *testing.T) {
	readings := []decode.Reading{
		{Name: "PipelineJSON", Type: "string", Value: `{"objects":[{"id":1}]}`},
		{Name: "InferLatencyMs", Type: "float64", Value: 2.5},
	}
	j := &Job{cfg: Config{PublishMode: PublishNonEmpty, PublishReadings: []string{"InferLatencyMs"}}}
	got := j.filterForPublish(readings)
	if len(got) != 1 || got[0].Name != "InferLatencyMs" {
		t.Fatalf("nonempty content must be evaluated before the whitelist: %v", got)
	}
}

func TestPublishPolicyDoesNotFilterCachedReadings(t *testing.T) {
	readings := []decode.Reading{{Name: "PipelineJSON", Type: "string", Value: `{"objects":[]}`}}
	j := &Job{
		cfg:      Config{PublishMode: PublishNone, PublishReadings: []string{"InferLatencyMs"}},
		stats:    &Stats{State: StateApplying},
		lastVals: map[string]any{},
	}
	j.cacheReadings(readings)
	if got := j.filterForPublish(readings); got != nil {
		t.Fatalf("none mode emitted readings: %v", got)
	}
	if got, ok := j.LastValue("PipelineJSON"); !ok || got != readings[0].Value {
		t.Fatalf("async publish policy changed the GET cache: value=%v, present=%v", got, ok)
	}
}

func TestFilterForPublishWhitelist(t *testing.T) {
	readings := []decode.Reading{
		{Name: "DetectionCount", Type: "int32", Value: int32(1)},
		{Name: "DetectionJSON", Type: "string", Value: `[{"label":"car"}]`},
		{Name: "Top1Label", Type: "string", Value: "car"},
		{Name: "InferLatencyMs", Type: "float64", Value: 1.2},
	}
	j := &Job{
		cfg: Config{
			PublishMode:     PublishAll,
			PublishReadings: []string{"DetectionJSON"},
		},
	}
	got := j.filterForPublish(readings)
	if len(got) != 1 || got[0].Name != "DetectionJSON" {
		t.Fatalf("whitelist must keep only listed readings, got %v", got)
	}
}
