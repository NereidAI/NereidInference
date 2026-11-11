// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"bytes"
	"context"
	"fmt"
	"image/jpeg"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"time"
)

const (
	// maxFrameBytes bounds one JPEG image supplied by an HTTP source.
	maxFrameBytes  = 32 * 1024 * 1024
	mpjpegIdleTime = 30 * time.Second
)

// snapshotPollInterval is the GET cadence for single-snapshot endpoints;
// per-job FPS sampling happens downstream, so the source rate only needs to
// be comfortably above typical job rates.
const snapshotPollInterval = 200 * time.Millisecond

// multipartSource consumes a persistent multipart/x-mixed-replace MJPEG
// stream (native camera MJPEG or an external go2rtc stream.mjpeg URL).
type multipartSource struct {
	url    string
	client *http.Client
}

func (s *multipartSource) Kind() string        { return "mpjpeg" }
func (s *multipartSource) RedactedURL() string { return RedactUserInfo(s.url) }

// Open reads parts until the stream ends; the Manager handles reconnect.
func (s *multipartSource) Open(ctx context.Context) (<-chan Frame, error) {
	out := make(chan Frame, 8)
	go func() {
		defer close(out)
		backoff := time.Second
		for ctx.Err() == nil {
			err := s.runOnce(ctx, out)
			if ctx.Err() != nil {
				return
			}
			// Reconnect after both errors and clean EOF; otherwise a camera
			// that closes idle MJPEG sockets immediately causes a hot loop.
			if sleepCtx(ctx, backoff) {
				return
			}
			if err != nil {
				backoff = nextBackoff(backoff, 30*time.Second)
			} else {
				backoff = time.Second
			}
		}
	}()
	return out, nil
}

func (s *multipartSource) runOnce(ctx context.Context, out chan<- Frame) error {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(streamCtx, http.MethodGet, s.url, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("mjpeg source %s returned HTTP %d", s.RedactedURL(), resp.StatusCode)
	}
	mediaType, params, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil {
		return fmt.Errorf("mjpeg content type: %w", err)
	}
	if mediaType != "multipart/x-mixed-replace" || params["boundary"] == "" {
		return fmt.Errorf("mjpeg source returned content type %q", mediaType)
	}
	body := &idleReader{r: resp.Body, timeout: mpjpegIdleTime, cancel: cancel}
	mr := multipart.NewReader(body, params["boundary"])
	for {
		part, err := mr.NextPart()
		if err != nil {
			return err
		}
		b, err := io.ReadAll(io.LimitReader(part, maxFrameBytes+1))
		part.Close()
		if err != nil {
			return err
		}
		if int64(len(b)) > maxFrameBytes {
			return fmt.Errorf("mjpeg part exceeds %d bytes", maxFrameBytes)
		}
		f, err := jpegFrame(b)
		if err != nil {
			continue
		}
		select {
		case out <- f:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// jpegSource polls an HTTP snapshot URL.
type jpegSource struct {
	url    string
	client *http.Client
}

func (s *jpegSource) Kind() string        { return "jpeg" }
func (s *jpegSource) RedactedURL() string { return RedactUserInfo(s.url) }

// Open polls the snapshot URL at snapshotPollInterval.
func (s *jpegSource) Open(ctx context.Context) (<-chan Frame, error) {
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

func (s *jpegSource) fetch(ctx context.Context) (Frame, error) {
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
		return Frame{}, fmt.Errorf("jpeg source returned HTTP %d", resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxFrameBytes+1))
	if err != nil {
		return Frame{}, err
	}
	if int64(len(b)) > maxFrameBytes {
		return Frame{}, fmt.Errorf("jpeg source response exceeds %d bytes", maxFrameBytes)
	}
	return jpegFrame(b)
}

func jpegFrame(b []byte) (Frame, error) {
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(b))
	if err != nil {
		return Frame{}, err
	}
	if err := validateGeometry(cfg.Width, cfg.Height); err != nil {
		return Frame{}, err
	}
	return Frame{Encoding: JPEG, Width: cfg.Width, Height: cfg.Height, Bytes: append([]byte(nil), b...), Time: time.Now()}, nil
}

// idleReader cancels the owning request context if no response bytes arrive
// within timeout, allowing a half-open TCP MJPEG connection to reconnect.
type idleReader struct {
	r       io.Reader
	timeout time.Duration
	cancel  context.CancelFunc
	timer   *time.Timer
}

func (r *idleReader) Read(p []byte) (int, error) {
	if r.timer == nil {
		r.timer = time.AfterFunc(r.timeout, r.cancel)
	} else {
		r.timer.Reset(r.timeout)
	}
	n, err := r.r.Read(p)
	if r.timer != nil {
		r.timer.Stop()
	}
	return n, err
}
