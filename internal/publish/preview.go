// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package publish hosts the local preview server and frame annotation. It
// is intentionally operator-facing diagnostics only: it streams the exact
// frame fed to inference with the decoder's own overlay (boxes, masks,
// keypoints, captions), never business rendering.
package publish

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/edgexfoundry/device-inference/internal/decode"
	"github.com/edgexfoundry/device-inference/internal/frame"
)

// Frame is one preview update: acquired frame plus its decoded result.
type Frame struct {
	Source   *frame.Frame
	Result   decode.Result
	Labels   []string
	Annotate bool
}

// Hub is the per-device latest-frame registry.
type Hub struct {
	mu     sync.Mutex
	frames map[string]*Frame
	subs   map[string]map[*hubSub]struct{}
}

type hubSub struct {
	ch   chan *Frame
	once sync.Once
}

func newHubSub() *hubSub {
	return &hubSub{ch: make(chan *Frame, 1)}
}

func (s *hubSub) close() {
	s.once.Do(func() { close(s.ch) })
}

// NewHub creates an empty preview hub.
func NewHub() *Hub {
	return &Hub{
		frames: map[string]*Frame{},
		subs:   map[string]map[*hubSub]struct{}{},
	}
}

// Update publishes a new frame; with no subscribers it only refreshes the
// snapshot, so idle operators cost no JPEG encoding.
func (h *Hub) Update(device string, f *Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.frames[device] = f
	for sub := range h.subs[device] {
		select {
		case sub.ch <- f:
		default:
			select {
			case <-sub.ch:
			default:
			}
			select {
			case sub.ch <- f:
			default:
			}
		}
	}
}

// Remove drops the retained snapshot and terminates HTTP streams for a
// deleted device.
func (h *Hub) Remove(device string) {
	h.mu.Lock()
	delete(h.frames, device)
	subs := h.subs[device]
	delete(h.subs, device)
	h.mu.Unlock()
	for sub := range subs {
		sub.close()
	}
}

func (h *Hub) subscribe(device string) *hubSub {
	sub := newHubSub()
	h.mu.Lock()
	if h.subs[device] == nil {
		h.subs[device] = map[*hubSub]struct{}{}
	}
	h.subs[device][sub] = struct{}{}
	if snapshot := h.frames[device]; snapshot != nil {
		sub.ch <- snapshot
	}
	h.mu.Unlock()
	return sub
}

func (h *Hub) unsubscribe(device string, sub *hubSub) {
	h.mu.Lock()
	if subs, ok := h.subs[device]; ok {
		delete(subs, sub)
	}
	h.mu.Unlock()
	sub.close()
}

// snapshot returns the latest frame for one device.
func (h *Hub) snapshot(device string) (*Frame, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	f, ok := h.frames[device]
	return f, ok
}

// Server exposes MJPEG and JPEG preview endpoints.
type Server struct {
	hub *Hub
	ln  net.Listener
	http.Server
}

// NewServer builds and binds the preview HTTP server.
func NewServer(bind string, hub *Hub) (*Server, error) {
	ln, err := net.Listen("tcp", bind)
	if err != nil {
		return nil, err
	}
	s := &Server{hub: hub, ln: ln}
	mux := http.NewServeMux()
	mux.HandleFunc("/preview/", s.handle)
	s.Server = http.Server{
		Addr:              bind,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	return s, nil
}

// Hub returns the frame registry served by this HTTP server.
func (s *Server) Hub() *Hub { return s.hub }

// Serve runs the already-bound HTTP server.
func (s *Server) Serve() error { return s.Server.Serve(s.ln) }

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	// Paths: /preview/{device}.mjpeg or /preview/{device}/frame.jpeg
	tail := r.URL.Path[len("/preview/"):]
	device, kind := splitTail(tail)
	if device == "" {
		http.NotFound(w, r)
		return
	}
	annotate := r.URL.Query().Get("annotate") != "0"
	switch kind {
	case "mjpeg":
		s.serveMJPEG(w, r, device, annotate)
	case "jpeg":
		s.serveJPEG(w, device, annotate)
	default:
		http.NotFound(w, r)
	}
}

func splitTail(tail string) (device, kind string) {
	if i := lastSlash(tail); i >= 0 {
		name, suffix := tail[:i], tail[i+1:]
		if suffix == "frame.jpeg" {
			return name, "jpeg"
		}
	}
	if len(tail) > len(".mjpeg") && tail[len(tail)-len(".mjpeg"):] == ".mjpeg" {
		return tail[:len(tail)-len(".mjpeg")], "mjpeg"
	}
	return "", ""
}

func lastSlash(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '/' {
			return i
		}
	}
	return -1
}

func (s *Server) serveJPEG(w http.ResponseWriter, device string, annotate bool) {
	f, ok := s.hub.snapshot(device)
	if !ok {
		http.Error(w, "no frame yet", http.StatusServiceUnavailable)
		return
	}
	b, err := EncodeJPEG(f, annotate)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "image/jpeg")
	_, _ = w.Write(b)
}

func (s *Server) serveMJPEG(w http.ResponseWriter, r *http.Request, device string, annotate bool) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frameboundary")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	sub := s.hub.subscribe(device)
	defer s.hub.unsubscribe(device, sub)
	for {
		select {
		case <-r.Context().Done():
			return
		case f, ok := <-sub.ch:
			if !ok {
				return
			}
			b, err := EncodeJPEG(f, annotate)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "--frameboundary\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(b))
			if _, err := w.Write(b); err != nil {
				return
			}
			if _, err := w.Write([]byte("\r\n")); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

// EncodeJPEG converts an acquired frame to JPEG, optionally after overlaying
// the decoded result. It is shared by the live MJPEG preview and binary EdgeX
// evidence readings so operators see the same annotated image that is stored.
func EncodeJPEG(f *Frame, annotate bool) ([]byte, error) {
	if f == nil || f.Source == nil {
		return nil, fmt.Errorf("empty preview frame")
	}
	canvas, err := toRGBA(f.Source)
	if err != nil {
		return nil, err
	}
	if annotate && f.Annotate && f.Result != nil {
		f.Result.Draw(canvas, f.Labels)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, canvas, &jpeg.Options{Quality: 80}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func toRGBA(f *frame.Frame) (*image.RGBA, error) {
	if f.Encoding == frame.RGB24 {
		if f.Width <= 0 || f.Height <= 0 || len(f.Bytes) < f.Width*f.Height*3 {
			return nil, fmt.Errorf("invalid rgb24 frame %dx%d (%d bytes)", f.Width, f.Height, len(f.Bytes))
		}
		img := image.NewRGBA(image.Rect(0, 0, f.Width, f.Height))
		for i := 0; i < f.Width*f.Height; i++ {
			img.Pix[4*i] = f.Bytes[3*i]
			img.Pix[4*i+1] = f.Bytes[3*i+1]
			img.Pix[4*i+2] = f.Bytes[3*i+2]
			img.Pix[4*i+3] = 255
		}
		return img, nil
	}
	return decodeJPEG(f.Bytes)
}

func decodeJPEG(b []byte) (*image.RGBA, error) {
	img, err := jpeg.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	rgba, ok := img.(*image.RGBA)
	if ok {
		return rgba, nil
	}
	out := image.NewRGBA(img.Bounds())
	drawRGBA(out, img)
	return out, nil
}

func drawRGBA(dst *image.RGBA, src image.Image) {
	b := src.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			dst.Set(x-b.Min.X, y-b.Min.Y, src.At(x, y))
		}
	}
}
