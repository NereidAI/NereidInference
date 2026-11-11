// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// newSource builds the Source implementation for a resolved reference.
func newSource(ref ResolvedRef, ffmpegCfg FFmpegConfig, log Logger) (Source, error) {
	log = loggerOrNop(log)
	if ref.DecodeAccel == DecodeAccelVAAPI {
		switch ref.Kind {
		case "rtsp", "rtsp-udp":
		default:
			return nil, fmt.Errorf("DecodeAccel=vaapi is supported only by rtsp sources, got kind %q", ref.Kind)
		}
	}
	switch ref.Kind {
	case "rtsp", "rtsp-udp":
		return &ffmpegSource{ref: ref, cfg: ffmpegCfg, log: log}, nil
	case "onvif-snapshot":
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.ResponseHeaderTimeout = 10 * time.Second
		return &edgexSnapshotSource{url: ref.URL, client: &http.Client{Timeout: 15 * time.Second, Transport: transport}}, nil
	case "mpjpeg":
		// The response is deliberately long-lived; idleReader bounds stalls
		// after headers arrive, while ResponseHeaderTimeout bounds connect.
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.ResponseHeaderTimeout = 10 * time.Second
		return &multipartSource{url: ref.URL, client: &http.Client{Transport: transport}}, nil
	case "jpeg":
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.ResponseHeaderTimeout = 10 * time.Second
		return &jpegSource{url: ref.URL, client: &http.Client{Timeout: 10 * time.Second, Transport: transport}}, nil
	default:
		return nil, fmt.Errorf("unsupported frame source kind %q", ref.Kind)
	}
}

// RedactUserInfo strips credentials from a URL for safe logging.
func RedactUserInfo(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.User == nil {
		return raw
	}
	u.User = url.User("***")
	return u.String()
}
