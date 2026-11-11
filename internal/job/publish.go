// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package job

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/edgexfoundry/device-inference/internal/decode"
)

// housekeepingReadings are produced by the service itself rather than by the
// model decoder; they never count as business content for nonempty filtering.
var housekeepingReadings = map[string]struct{}{
	"InferLatencyMs": {},
	"FrameJPEG":      {},
}

// filterForPublish applies the binding-level telemetry policy: PublishMode
// none/nonempty/all plus an optional reading-name whitelist. Cached values and
// the local preview are unaffected — only bus emission is filtered.
func (j *Job) filterForPublish(readings []decode.Reading) []decode.Reading {
	if j.cfg.PublishMode == PublishNone {
		return nil
	}
	if j.cfg.PublishMode == PublishNonEmpty && !hasBusinessContent(readings) {
		return nil
	}
	if len(j.cfg.PublishReadings) == 0 {
		return readings
	}
	allowed := make(map[string]struct{}, len(j.cfg.PublishReadings))
	for _, name := range j.cfg.PublishReadings {
		allowed[name] = struct{}{}
	}
	out := make([]decode.Reading, 0, len(readings))
	for _, reading := range readings {
		if _, ok := allowed[reading.Name]; ok {
			out = append(out, reading)
		}
	}
	return out
}

// hasBusinessContent reports whether at least one decoder-produced reading
// carries substantive content. Empty structured payloads ("[]", "{}"), blank
// strings and zero *Count readings describe a frame the model found nothing
// on; scalar regressions and non-empty JSON always count as content.
func hasBusinessContent(readings []decode.Reading) bool {
	for _, reading := range readings {
		if _, ok := housekeepingReadings[reading.Name]; ok {
			continue
		}
		if !isEmptyBusinessReading(reading) {
			return true
		}
	}
	return false
}

func isEmptyBusinessReading(reading decode.Reading) bool {
	switch reading.Type {
	case "string":
		s := strings.TrimSpace(fmt.Sprint(reading.Value))
		if reading.Name == "PipelineJSON" {
			var document struct {
				Objects []json.RawMessage `json:"objects"`
			}
			if json.Unmarshal([]byte(s), &document) == nil && len(document.Objects) == 0 {
				return true
			}
		}
		return s == "" || s == "[]" || s == "{}"
	case "int32", "int64":
		if strings.HasSuffix(reading.Name, "Count") {
			return numericIsZero(reading.Value)
		}
		return false
	default:
		return false
	}
}

func numericIsZero(value any) bool {
	switch n := value.(type) {
	case int:
		return n == 0
	case int32:
		return n == 0
	case int64:
		return n == 0
	case float64:
		return n == 0
	default:
		return false
	}
}
