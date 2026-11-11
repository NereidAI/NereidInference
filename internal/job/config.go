// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package job orchestrates one inference pipeline per EdgeX device:
// trigger scheduling, frame acquisition, preprocessing, backend call,
// generic decoding, event emission, preview, backpressure and statistics.
package job

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/edgexfoundry/device-inference/internal/frame"
	"github.com/edgexfoundry/device-inference/internal/model"
	"github.com/edgexfoundry/device-inference/internal/modelserver"
)

// streamProfilePattern bounds ONVIF media ProfileTokens to the transport-safe
// character set shared with TargetDevice (no spaces, quotes, punctuation that
// could carry injection into command bodies).
var streamProfilePattern = regexp.MustCompile(`^[A-Za-z0-9_.:,-]+$`)

// renderNodePattern accepts either a render-node basename or its fixed
// /dev/dri path; cloud-supplied values must never be arbitrary paths.
var renderNodePattern = regexp.MustCompile(`^(?:/dev/dri/)?renderD\d+$`)

// ProtocolKey is the protocol properties section name.
const ProtocolKey = "Inference"

// Trigger modes.
const (
	TriggerContinuous = "continuous"
	TriggerPeriodic   = "periodic"
	TriggerEvent      = "event"
	TriggerCommand    = "command"
)

// Publish modes for the binding-level telemetry filter.
const (
	PublishAll      = "all"
	PublishNonEmpty = "nonempty"
	PublishNone     = "none"
)

// readingNamePattern bounds whitelisted EdgeX resource names.
var readingNamePattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]{0,63}$`)

// Seven days is far beyond normal camera polling but stays safely within
// time.Duration's int64-nanosecond range.
const maxScheduleMs = 7 * 24 * 60 * 60 * 1000

// Config is the flattened protocol-properties view of one job. EdgeX
// protocol properties are scalar k/v pairs, which is why all values arrive
// as strings.
type Config struct {
	SourceKind   string
	SourceDevice string
	SourceURL    string
	SecretName   string
	// StreamProfile is an ONVIF media ProfileToken selecting the encoded
	// stream (main/sub/third) for the onvif-camera federation source; empty
	// means the camera service default profile.
	StreamProfile string
	// DecodeAccel selects the ffmpeg decode pipeline for streaming sources:
	// "software" (default) or "vaapi" (frame.DecodeAccel*). Snapshot and
	// other HTTP sources never start ffmpeg and must leave it at software.
	DecodeAccel string
	// VAAPIDevice optionally selects a job-specific DRM render node
	// ("renderD129" or "/dev/dri/renderD129"); empty uses the service
	// default. It permits choosing integrated vs discrete GPU per stream.
	VAAPIDevice string

	// PipelineID selects the NI-native multi-model path. The EdgeX property
	// spelling is intentionally PipelineId; PipelineID and ModelID/ModelVersion
	// are mutually exclusive admission forms.
	PipelineID   string
	ModelID      string
	ModelVersion string
	// ServableID is parsed only to reject unsupported external execution at
	// admission. It is never a fallback selector for a local catalog/graph.
	ServableID   string
	Endpoint     string
	TargetDevice string

	Trigger   string
	FPS       float64
	PeriodMs  int
	MaxIdleMs int

	ErrorThreshold int
	ConfThreshold  float64

	EmitFrame       bool
	AnnotatePreview bool
	PromptTemplate  string
	SamplingJSON    string
	MaxInferMs      int
	FFmpegExtra     string

	// PublishMode controls which per-frame events reach the EdgeX message bus:
	// "all" (default) emits every cycle, "nonempty" suppresses frames without
	// business content, "none" keeps local preview/caches only.
	PublishMode string
	// PublishReadings optionally whitelists the reading resource names sent
	// northbound. Empty means every profile-declared reading is eligible.
	PublishReadings []string
}

// FromProperties parses and validates the Inference protocol properties.
func FromProperties(props map[string]any) (Config, error) {
	c := Config{
		SourceKind:      str(props, "SourceKind", "onvif-camera"),
		SourceDevice:    str(props, "SourceDevice", ""),
		SourceURL:       str(props, "SourceURL", ""),
		SecretName:      str(props, "SecretName", ""),
		StreamProfile:   strings.TrimSpace(str(props, "StreamProfile", "")),
		DecodeAccel:     strings.ToLower(strings.TrimSpace(str(props, "DecodeAccel", ""))),
		VAAPIDevice:     strings.TrimSpace(str(props, "VAAPIDevice", "")),
		PipelineID:      str(props, "PipelineId", ""),
		ModelID:         str(props, "ModelID", ""),
		ModelVersion:    str(props, "ModelVersion", ""),
		ServableID:      str(props, "ServableID", ""),
		Endpoint:        str(props, "Endpoint", ""),
		TargetDevice:    str(props, "TargetDevice", "CPU"),
		Trigger:         str(props, "Trigger", TriggerContinuous),
		FPS:             num(props, "FPS", 5),
		PeriodMs:        int(num(props, "PeriodMs", 1000)),
		MaxIdleMs:       int(num(props, "MaxIdleMs", 0)),
		ErrorThreshold:  int(num(props, "ErrorThreshold", 5)),
		ConfThreshold:   num(props, "ConfThreshold", 0),
		EmitFrame:       boolean(props, "EmitFrame", false),
		AnnotatePreview: boolean(props, "AnnotatePreview", true),
		PromptTemplate:  str(props, "PromptTemplate", ""),
		SamplingJSON:    str(props, "Sampling", ""),
		MaxInferMs:      int(num(props, "MaxInferMs", 0)),
		FFmpegExtra:     str(props, "FFmpegExtra", ""),
		PublishMode:     strings.ToLower(str(props, "PublishMode", PublishAll)),
		PublishReadings: splitCSV(str(props, "PublishReadings", "")),
	}
	if c.ServableID != "" {
		return c, fmt.Errorf("%s: Inference.ServableID external execution is not supported", ReasonConfigInvalid)
	}
	hasModelSelection := c.ModelID != "" || c.ModelVersion != ""
	if c.PipelineID != "" && hasModelSelection {
		return c, fmt.Errorf("Inference.PipelineId is mutually exclusive with Inference.ModelID/ModelVersion")
	}
	if c.PipelineID == "" {
		if c.ModelID == "" {
			return c, fmt.Errorf("exactly one inference selection is required: Inference.PipelineId or Inference.ModelID (with optional ModelVersion)")
		}
	} else if !model.IsSafeName(c.PipelineID) {
		return c, fmt.Errorf("invalid Inference.PipelineId %q", c.PipelineID)
	}
	targetDevice, err := modelserver.NormalizeTargetDevice(c.TargetDevice)
	if err != nil {
		return c, err
	}
	c.TargetDevice = targetDevice
	switch c.Trigger {
	case TriggerContinuous, TriggerPeriodic, TriggerEvent, TriggerCommand:
	default:
		return c, fmt.Errorf("invalid Trigger %q", c.Trigger)
	}
	switch c.SourceKind {
	case "none", "onvif-camera", "onvif-snapshot", "rtsp", "rtsp-udp", "mpjpeg", "jpeg":
	default:
		return c, fmt.Errorf("invalid SourceKind %q", c.SourceKind)
	}
	if c.PipelineID != "" && c.SourceKind == "none" {
		return c, fmt.Errorf("SourceKind=none is not valid with Inference.PipelineId; an image source is required")
	}
	if c.SourceKind != "none" {
		if (c.SourceKind == "onvif-camera" || c.SourceKind == "onvif-snapshot") && c.SourceDevice == "" {
			return c, fmt.Errorf("SourceDevice is required for SourceKind %s", c.SourceKind)
		}
		if c.SourceKind != "onvif-camera" && c.SourceKind != "onvif-snapshot" && c.SourceURL == "" {
			return c, fmt.Errorf("SourceURL is required for SourceKind %q", c.SourceKind)
		}
	}
	// A ProfileToken is only meaningful for the federated onvif-camera
	// source; snapshot uses the Snapshot command, direct-URL sources use
	// their own path. Tokens stay transport-safe like every cloud-derived
	// string (same character set as TargetDevice, max 64).
	if c.StreamProfile != "" {
		if c.SourceKind != "onvif-camera" {
			return c, fmt.Errorf("StreamProfile is supported only with SourceKind=onvif-camera, got %s", c.SourceKind)
		}
		if len(c.StreamProfile) > 64 || !streamProfilePattern.MatchString(c.StreamProfile) {
			return c, fmt.Errorf("invalid StreamProfile %q", c.StreamProfile)
		}
	}
	// VA-API decode only applies to ffmpeg streaming sources; snapshot and
	// HTTP-JPEG sources have no ffmpeg child to attach the device to.
	switch c.DecodeAccel {
	case "", frame.DecodeAccelSoftware:
		c.DecodeAccel = frame.DecodeAccelSoftware
		if c.VAAPIDevice != "" {
			return c, fmt.Errorf("VAAPIDevice requires DecodeAccel=vaapi")
		}
	case frame.DecodeAccelVAAPI:
		// Both direct RTSP and federated onvif-camera streams run ffmpeg;
		// snapshot and HTTP-JPEG sources have no decoder child.
		switch c.SourceKind {
		case "rtsp", "rtsp-udp", "onvif-camera":
		default:
			return c, fmt.Errorf("DecodeAccel=vaapi is supported only with SourceKind rtsp, rtsp-udp or onvif-camera, got %s", c.SourceKind)
		}
		if c.VAAPIDevice != "" {
			if !renderNodePattern.MatchString(c.VAAPIDevice) {
				return c, fmt.Errorf("invalid VAAPIDevice %q", c.VAAPIDevice)
			}
			if !strings.HasPrefix(c.VAAPIDevice, "/dev/dri/") {
				c.VAAPIDevice = "/dev/dri/" + c.VAAPIDevice
			}
		}
	default:
		return c, fmt.Errorf("invalid DecodeAccel %q", c.DecodeAccel)
	}
	if math.IsNaN(c.FPS) || math.IsInf(c.FPS, 0) || c.FPS < 0 || c.FPS > 60 {
		return c, fmt.Errorf("FPS must be a finite number in 0..60, got %g", c.FPS)
	}
	if c.Trigger == TriggerPeriodic && (c.PeriodMs <= 0 || c.PeriodMs > maxScheduleMs) {
		return c, fmt.Errorf("PeriodMs must be in 1..%d for periodic trigger", maxScheduleMs)
	}
	if c.MaxInferMs < 0 || c.MaxInferMs > 10*60*1000 {
		return c, fmt.Errorf("MaxInferMs must be in 0..600000")
	}
	if c.MaxIdleMs > maxScheduleMs {
		return c, fmt.Errorf("MaxIdleMs must be in 0..%d", maxScheduleMs)
	}
	if c.ErrorThreshold < 0 {
		return c, fmt.Errorf("ErrorThreshold must not be negative")
	}
	if math.IsNaN(c.ConfThreshold) || math.IsInf(c.ConfThreshold, 0) ||
		c.ConfThreshold < 0 || c.ConfThreshold > 1 {
		return c, fmt.Errorf("ConfThreshold must be a finite number in 0..1, got %g", c.ConfThreshold)
	}
	if c.MaxIdleMs < 0 {
		return c, fmt.Errorf("MaxIdleMs must not be negative")
	}
	if c.SamplingJSON != "" && !json.Valid([]byte(c.SamplingJSON)) {
		return c, fmt.Errorf("sampling must be valid JSON")
	}
	if c.FFmpegExtra != "" {
		if err := frame.ValidateFFmpegExtra(c.FFmpegExtra); err != nil {
			return c, err
		}
	}
	switch c.PublishMode {
	case PublishAll, PublishNonEmpty, PublishNone:
	default:
		return c, fmt.Errorf("invalid PublishMode %q", c.PublishMode)
	}
	seenReadings := map[string]struct{}{}
	for _, name := range c.PublishReadings {
		if !readingNamePattern.MatchString(name) {
			return c, fmt.Errorf("invalid PublishReadings resource name %q", name)
		}
		if _, dup := seenReadings[name]; dup {
			return c, fmt.Errorf("duplicate PublishReadings resource name %q", name)
		}
		seenReadings[name] = struct{}{}
	}
	return c, nil
}

// splitCSV parses a comma-separated protocol-property value into trimmed,
// de-duplicated tokens, dropping empty entries.
func splitCSV(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]struct{}{}
	for _, part := range parts {
		token := strings.TrimSpace(part)
		if token == "" {
			continue
		}
		if _, ok := seen[token]; ok {
			continue
		}
		seen[token] = struct{}{}
		out = append(out, token)
	}
	return out
}

func str(m map[string]any, key, def string) string {
	if v, ok := m[key]; ok {
		switch t := v.(type) {
		case string:
			return strings.TrimSpace(t)
		case fmt.Stringer:
			return t.String()
		default:
			return strings.TrimSpace(fmt.Sprintf("%v", t))
		}
	}
	return def
}

func num(m map[string]any, key string, def float64) float64 {
	if v, ok := m[key]; ok {
		switch t := v.(type) {
		case float64:
			return t
		case float32:
			return float64(t)
		case int:
			return float64(t)
		case int64:
			return float64(t)
		case string:
			if n, err := strconv.ParseFloat(strings.TrimSpace(t), 64); err == nil {
				return n
			}
		}
	}
	return def
}

func boolean(m map[string]any, key string, def bool) bool {
	if v, ok := m[key]; ok {
		switch t := v.(type) {
		case bool:
			return t
		case string:
			b, err := strconv.ParseBool(strings.TrimSpace(t))
			if err == nil {
				return b
			}
		}
	}
	return def
}
