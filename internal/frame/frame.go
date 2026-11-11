// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package frame provides media frame acquisition. A Source produces
// decoded frames (raw RGB24 or already-decoded JPEG bytes) from one URL; the
// Manager de-duplicates sources by URL so that N jobs bound to one camera
// share a single decoder and receive an in-process fan-out.
//
// The DS never opens RTSP or decodes H.264 in Go: the rtsp source kind runs a
// templated, unprivileged ffmpeg child process and reads fixed-size RGB
// frames from its stdout pipe. The mpjpeg/jpeg kinds are pure standard
// library. See implementation design section 4.
package frame

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Encoding identifies the pixel encoding of Frame.Bytes.
type Encoding string

const (
	// RGB24 is tightly packed width*height*3 byte rows, R first.
	RGB24 Encoding = "rgb24"
	// JPEG is a complete JPEG image; preprocessing decodes it on demand.
	JPEG Encoding = "jpeg"
)

// maxFramePixels bounds decoded camera geometry. A compressed JPEG or RTSP
// header can otherwise advertise tens of thousands of pixels and force
// multi-gigabyte RGB allocations.
const maxFramePixels = 7680 * 4320

func validateGeometry(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("invalid frame geometry %dx%d", width, height)
	}
	// Compare without multiplying first: width and height come from external
	// media metadata, so the product could overflow on 32-bit systems.
	if width > maxFramePixels/height {
		return fmt.Errorf("frame geometry %dx%d exceeds %d pixels", width, height, maxFramePixels)
	}
	return nil
}

// Frame is one acquired picture. The producer allocates a fresh slice per
// frame; subscribers must not retain Bytes beyond processing without copy.
type Frame struct {
	Encoding Encoding
	Width    int
	Height   int
	Bytes    []byte
	// Time is when the frame was produced by the source.
	Time time.Time
}

// Logger is the small logging surface frame sources need. A nil logger
// supplied to the constructors is replaced with a no-op implementation.
type Logger interface {
	Infof(format string, args ...any)
	Warnf(format string, args ...any)
	Debugf(format string, args ...any)
}

type nopLogger struct{}

func (nopLogger) Infof(string, ...any)  {}
func (nopLogger) Warnf(string, ...any)  {}
func (nopLogger) Debugf(string, ...any) {}

func loggerOrNop(log Logger) Logger {
	if log == nil {
		return nopLogger{}
	}
	return log
}

// Source is one frame producer bound to one URL.
type Source interface {
	// Open starts acquisition and returns the frame channel, which is closed
	// when ctx is canceled or acquisition fails permanently.
	Open(ctx context.Context) (<-chan Frame, error)
	// Kind reports the source kind (rtsp, mpjpeg, jpeg).
	Kind() string
	// RedactedURL returns the URL without user information for logging.
	RedactedURL() string
}

// Decode acceleration selectors carried by ResolvedRef.
const (
	// DecodeAccelSoftware keeps H.264/H.265 decode and scaling in
	// libavcodec on the CPU.
	DecodeAccelSoftware = "software"
	// DecodeAccelVAAPI decodes onto a VA-API DRM render node; scaling runs
	// on the same device and frames are downloaded as NV12 before the RGB24
	// conversion. A missing device or driver fails the child at startup;
	// ffmpeg's silent software fallback is never used.
	DecodeAccelVAAPI = "vaapi"
)

// ResolvedRef is everything the Manager needs to open a source.
type ResolvedRef struct {
	Kind string // rtsp|rtsp-udp|mpjpeg|jpeg
	URL  string
	// ExtraArgs are optional extra ffmpeg argv tokens for rtsp sources.
	ExtraArgs string
	// DecodeAccel is software (default) or vaapi; rtsp kinds only.
	DecodeAccel string
	// VAAPIDevice is an optional job-specific DRM render node; empty uses
	// the service default FFmpeg.VAAPIDevice.
	VAAPIDevice string
}

// Subscription is one job's fan-out handle.
type Subscription struct {
	// C receives frames while the subscription is active. Buffer is 1 and
	// slow consumers drop the oldest frame rather than blocking the source.
	C        <-chan Frame
	ch       chan Frame
	parent   *Manager
	releaseO sync.Once
	released chan struct{}
}

// Release unsubscribes; when the last subscriber is gone the underlying
// source process/connection is stopped. It is safe to call repeatedly.
func (s *Subscription) Release() {
	s.releaseO.Do(func() {
		close(s.released)
		s.parent.release(s)
	})
}

// Manager owns the URL-deduplicated source table.
type Manager struct {
	mu        sync.Mutex
	sources   map[string]*managedSource
	ffmpegCfg FFmpegConfig
	log       Logger
}

// FFmpegConfig parameterizes the ffmpeg child process; zero values are
// replaced by package defaults before use.
type FFmpegConfig struct {
	Bin               string
	ProbeBin          string
	Global            string
	ConnectTimeoutSec int
	InputTemplates    map[string]string
	OutputTemplate    string
	ExtraArgs         string
	// AuthChallengeNormalize runs RTSP sources through a loopback relay
	// that collapses a camera's combined Basic+Digest 401 challenge to the
	// single Basic challenge libavformat can answer. TCP rtsp kind only.
	AuthChallengeNormalize bool
	DropUID                int
	DropGID                int
	// VAAPIDevice is the DRM render node ffmpeg opens for jobs that select
	// DecodeAccel=vaapi. An unreachable node makes -init_hw_device fail
	// before the input opens, so a misconfigured site fails loudly instead
	// of silently burning CPU.
	VAAPIDevice string
}

// NewManager creates an empty source Manager using cfg for rtsp sources.
func NewManager(cfg FFmpegConfig, log Logger) (*Manager, error) {
	cfg = mergeFFmpegConfig(cfg)
	if err := validateFFmpegConfig(cfg); err != nil {
		return nil, err
	}
	return &Manager{
		sources:   make(map[string]*managedSource),
		ffmpegCfg: cfg,
		log:       loggerOrNop(log),
	}, nil
}

func validateFFmpegConfig(cfg FFmpegConfig) error {
	if err := validateExtraArgs(cfg.ExtraArgs); err != nil {
		return fmt.Errorf("FFmpeg.ExtraArgs: %w", err)
	}
	for name, tmpl := range cfg.InputTemplates {
		if !strings.Contains(tmpl, "{url}") {
			return fmt.Errorf("FFmpeg input template %q must contain {url}", name)
		}
	}
	return nil
}

func mergeFFmpegConfig(in FFmpegConfig) FFmpegConfig {
	def := DefaultFFmpegConfig()
	if in.Bin == "" {
		in.Bin = def.Bin
	}
	if in.ProbeBin == "" {
		in.ProbeBin = def.ProbeBin
	}
	if in.Global == "" {
		in.Global = def.Global
	}
	if in.ConnectTimeoutSec <= 0 {
		in.ConnectTimeoutSec = def.ConnectTimeoutSec
	}
	if in.OutputTemplate == "" {
		in.OutputTemplate = def.OutputTemplate
	}
	if in.InputTemplates == nil {
		in.InputTemplates = map[string]string{}
	}
	for key, value := range def.InputTemplates {
		if in.InputTemplates[key] == "" {
			in.InputTemplates[key] = value
		}
	}
	if in.DropUID <= 0 {
		in.DropUID = def.DropUID
	}
	if in.DropGID <= 0 {
		in.DropGID = def.DropGID
	}
	if in.VAAPIDevice == "" {
		in.VAAPIDevice = def.VAAPIDevice
	}
	return in
}

// DefaultFFmpegConfig returns the built-in ffmpeg argv templates. Sites
// parse the templates; there is no shell and therefore no injection surface.
func DefaultFFmpegConfig() FFmpegConfig {
	return FFmpegConfig{
		Bin:               "ffmpeg",
		ProbeBin:          "ffprobe",
		Global:            "-hide_banner -loglevel warning",
		ConnectTimeoutSec: 5,
		// Input-side low-latency options (must precede -i and stay valid for
		// both ffmpeg and the ffprobe geometry probe, which share this
		// template):
		//   -flags low_delay     decoder: no B-frame reorder buffering
		//   -max_delay 500000    cap demuxer jitter/reorder delay at 500ms
		//   -probesize/-analyzeduration  bound startup/reconnect probing
		// Deliberately not used: -fflags nobuffer (measured to throttle this
		// RTSP/TCP camera from 25fps to ~6fps) and -thread_queue_size (not
		// accepted by ffprobe 6.1.2, and the rawvideo pipe is drained at
		// full speed so the demux queue never accumulates anyway).
		InputTemplates: map[string]string{
			"rtsp":     "-flags low_delay -max_delay 500000 -probesize 262144 -analyzeduration 1000000 -rtsp_transport tcp -timeout {timeout_us} -i {url}",
			"rtsp-udp": "-flags low_delay -max_delay 500000 -probesize 262144 -analyzeduration 1000000 -rtsp_transport udp -timeout {timeout_us} -i {url}",
		},
		OutputTemplate: "-an -f rawvideo -pix_fmt rgb24 pipe:1",
		// The relay is a compatibility shim for multi-scheme 401
		// challenges; service configuration may disable it.
		AuthChallengeNormalize: true,
		DropUID:                1000,
		DropGID:                1000,
		// First DRM render node, the conventional integrated-GPU node on
		// Intel/AMD Linux hosts; discrete or multi-GPU nodes are
		// operator-tuned through FFmpeg.VAAPIDevice.
		VAAPIDevice: "/dev/dri/renderD128",
	}
}

type managedSource struct {
	ref     ResolvedRef
	src     Source
	mu      sync.Mutex
	subs    map[*Subscription]struct{}
	cancel  context.CancelFunc
	runDone chan struct{}
}

// Subscribe returns a fan-out subscription for the resolved source. The
// first subscriber starts acquisition; releasing the last one stops it.
func (m *Manager) Subscribe(parentCtx context.Context, ref ResolvedRef) (*Subscription, error) {
	switch ref.DecodeAccel {
	case "", DecodeAccelSoftware, DecodeAccelVAAPI:
	default:
		return nil, fmt.Errorf("unsupported DecodeAccel %q", ref.DecodeAccel)
	}
	if ref.DecodeAccel == "" {
		ref.DecodeAccel = DecodeAccelSoftware
	}
	m.mu.Lock()
	ms, ok := m.sources[ref.URL]
	if ok && (ms.ref.Kind != ref.Kind || ms.ref.ExtraArgs != ref.ExtraArgs ||
		ms.ref.DecodeAccel != ref.DecodeAccel || ms.ref.VAAPIDevice != ref.VAAPIDevice) {
		m.mu.Unlock()
		return nil, fmt.Errorf("existing source for %s uses conflicting source settings", RedactUserInfo(ref.URL))
	}
	if !ok {
		src, err := newSource(ref, m.ffmpegCfg, m.log)
		if err != nil {
			m.mu.Unlock()
			return nil, err
		}
		ctx, cancel := context.WithCancel(context.Background())
		ms = &managedSource{
			ref:     ref,
			src:     src,
			subs:    map[*Subscription]struct{}{},
			cancel:  cancel,
			runDone: make(chan struct{}),
		}
		frames, err := src.Open(ctx)
		if err != nil {
			cancel()
			m.mu.Unlock()
			return nil, err
		}
		m.sources[ref.URL] = ms
		go m.fanOut(ms, frames)
	}
	ch := make(chan Frame, 1)
	sub := &Subscription{C: ch, ch: ch, parent: m, released: make(chan struct{})}
	ms.mu.Lock()
	ms.subs[sub] = struct{}{}
	ms.mu.Unlock()
	m.mu.Unlock()

	// Parent cancellation releases the subscription automatically so device
	// removal never leaks a decoder.
	go func() {
		select {
		case <-parentCtx.Done():
			sub.Release()
		case <-ms.runDone:
			sub.Release()
		case <-sub.released:
		}
	}()
	return sub, nil
}

func (m *Manager) release(s *Subscription) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for url, ms := range m.sources {
		ms.mu.Lock()
		if _, ok := ms.subs[s]; !ok {
			ms.mu.Unlock()
			continue
		}
		delete(ms.subs, s)
		close(s.ch)
		empty := len(ms.subs) == 0
		ms.mu.Unlock()
		if empty {
			ms.cancel()
			delete(m.sources, url)
		}
		return
	}
}

// fanOut delivers every frame to all current subscribers with drop-oldest
// semantics: one slow job must never stall decoding or other jobs.
func (m *Manager) fanOut(ms *managedSource, frames <-chan Frame) {
	defer close(ms.runDone)
	for f := range frames {
		ms.mu.Lock()
		for s := range ms.subs {
			select {
			case s.ch <- f:
			default:
				select {
				case <-s.ch:
				default:
				}
				select {
				case s.ch <- f:
				default:
				}
			}
		}
		ms.mu.Unlock()
	}
}

// StopAll cancels every source (used on DS shutdown).
func (m *Manager) StopAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, ms := range m.sources {
		ms.cancel()
	}
	m.sources = map[string]*managedSource{}
}
