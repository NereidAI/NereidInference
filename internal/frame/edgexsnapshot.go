// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// edgexSnapshotSource polls the EdgeX device-onvif-camera Snapshot command.
// That service performs camera-specific ONVIF/HTTP digest authentication and
// returns the image inside an EdgeX event's Binary reading. This source is
// useful when a camera's ONVIF Media profile works but its RTSP server uses an
// authentication variant unsupported by ffmpeg.
type edgexSnapshotSource struct {
	url    string
	client *http.Client
}

type edgexSnapshotResponse struct {
	Event struct {
		Readings []struct {
			ValueType   string `json:"valueType"`
			MediaType   string `json:"mediaType"`
			BinaryValue []byte `json:"binaryValue"`
		} `json:"readings"`
	} `json:"event"`
	Message string `json:"message"`
}

func (s *edgexSnapshotSource) Kind() string        { return "onvif-snapshot" }
func (s *edgexSnapshotSource) RedactedURL() string { return s.url }

// Open polls the command URL at snapshotPollInterval. The source rate is kept
// comfortably above typical job FPS values; each job samples the fan-out feed.
func (s *edgexSnapshotSource) Open(ctx context.Context) (<-chan Frame, error) {
	out := make(chan Frame, 8)
	go func() {
		defer close(out)
		ticker := time.NewTicker(snapshotPollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				f, err := s.fetch(ctx)
				if err != nil {
					continue
				}
				select {
				case out <- f:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return out, nil
}

func (s *edgexSnapshotSource) fetch(ctx context.Context) (Frame, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return Frame{}, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return Frame{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Frame{}, fmt.Errorf("ONVIF snapshot command returned HTTP %d", resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxFrameBytes+1))
	if err != nil {
		return Frame{}, err
	}
	if int64(len(b)) > maxFrameBytes {
		return Frame{}, fmt.Errorf("ONVIF snapshot response exceeds %d bytes", maxFrameBytes)
	}
	var payload edgexSnapshotResponse
	if err := json.Unmarshal(b, &payload); err != nil {
		return Frame{}, fmt.Errorf("decode ONVIF snapshot event: %w", err)
	}
	for _, reading := range payload.Event.Readings {
		if reading.ValueType != "Binary" || len(reading.BinaryValue) == 0 {
			continue
		}
		return jpegFrame(reading.BinaryValue)
	}
	if payload.Message != "" {
		return Frame{}, fmt.Errorf("ONVIF snapshot command failed: %s", payload.Message)
	}
	return Frame{}, fmt.Errorf("ONVIF snapshot event contains no binary JPEG reading")
}
