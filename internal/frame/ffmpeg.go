// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ffmpegSource runs one ffmpeg child per RTSP URL and emits fixed-size
// RGB24 frames on stdout. Reconnection uses capped exponential backoff;
// credential handling and codec quirks stay inside ffmpeg.
type ffmpegSource struct {
	ref        ResolvedRef
	cfg        FFmpegConfig
	log        Logger
	normalizer *rtspAuthNormalizer
}

// Kind returns the source kind.
func (s *ffmpegSource) Kind() string { return s.ref.Kind }

// RedactedURL returns the URL without embedded credentials.
func (s *ffmpegSource) RedactedURL() string { return RedactUserInfo(s.ref.URL) }

// vaapiDevice resolves the render node for a hardware-decoded source.
func (s *ffmpegSource) vaapiDevice() string {
	if s.ref.VAAPIDevice != "" {
		return s.ref.VAAPIDevice
	}
	return s.cfg.VAAPIDevice
}

// probeResult is the subset of ffprobe JSON we consume.
type probeResult struct {
	Streams []struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"streams"`
}

func (s *ffmpegSource) probe(ctx context.Context) (int, int, error) {
	key := s.ref.Kind
	inputTemplate := s.cfg.InputTemplates[key]
	if inputTemplate == "" {
		inputTemplate = s.cfg.InputTemplates["rtsp"]
	}
	// ffprobe must use the same transport/socket-timeout options as ffmpeg.
	args := tokenize("-v error")
	args = append(args, renderTemplate(inputTemplate, s.inputURL(), s.cfg)...)
	args = append(args, tokenize("-select_streams v:0 -print_format json -show_streams")...)
	probeCtx, cancel := context.WithTimeout(ctx, time.Duration(maxInt(s.cfg.ConnectTimeoutSec, 1)+5)*time.Second)
	defer cancel()
	cmd := exec.CommandContext(probeCtx, s.cfg.ProbeBin, args...) //nolint:gosec // argv is tokenized from service configuration, never evaluated by a shell.
	var out bytes.Buffer
	cmd.Stdout = &out
	stderr := s.captureStderr("ffprobe", cmd)
	s.configureProc(cmd)
	if err := cmd.Run(); err != nil {
		s.log.Warnf("ffprobe failed for %s: %v: %s", s.RedactedURL(), err, stderr.firstLine())
		return 0, 0, fmt.Errorf("ffprobe %s: %w", s.RedactedURL(), err)
	}
	var pr probeResult
	if err := json.Unmarshal(out.Bytes(), &pr); err != nil {
		return 0, 0, err
	}
	for _, st := range pr.Streams {
		if st.Width <= 0 || st.Height <= 0 {
			continue
		}
		if err := validateGeometry(st.Width, st.Height); err != nil {
			return 0, 0, err
		}
		return st.Width, st.Height, nil
	}
	return 0, 0, fmt.Errorf("ffprobe found no video stream")
}

// Open starts the reconnect loop and returns the frame channel.
func (s *ffmpegSource) Open(ctx context.Context) (<-chan Frame, error) {
	if err := validateExtraArgs(s.ref.ExtraArgs); err != nil {
		return nil, err
	}
	// TCP-only: rtsp-udp media bypasses the loopback relay and must connect
	// to the camera directly.
	if s.ref.Kind == "rtsp" && s.cfg.AuthChallengeNormalize {
		normalizer, err := startRTSPAuthNormalizer(s.ref.URL, s.log)
		if err != nil {
			return nil, fmt.Errorf("start rtsp auth normalizer for %s: %w", s.RedactedURL(), err)
		}
		s.normalizer = normalizer
	}
	out := make(chan Frame, 8)
	go func() {
		defer close(out)
		if s.normalizer != nil {
			defer func() { _ = s.normalizer.Close() }()
		}
		backoff := time.Second
		const maxBackoff = 30 * time.Second
		for ctx.Err() == nil {
			w, h, err := s.probe(ctx)
			if err != nil {
				if sleepCtx(ctx, backoff) {
					return
				}
				backoff = nextBackoff(backoff, maxBackoff)
				continue
			}
			n, err := s.runOnce(ctx, w, h, out)
			if ctx.Err() != nil {
				return
			}
			if err != nil || n == 0 {
				if sleepCtx(ctx, backoff) {
					return
				}
				backoff = nextBackoff(backoff, maxBackoff)
				continue
			}
			// A process that exits after only a few frames can otherwise
			// reconnect with no delay and trigger camera auth lockouts.
			if sleepCtx(ctx, time.Second) {
				return
			}
			backoff = time.Second
		}
	}()
	return out, nil
}

// runOnce runs a single ffmpeg process until EOF, error or context cancel.
// It returns the number of frames delivered.
func (s *ffmpegSource) runOnce(ctx context.Context, w, h int, out chan<- Frame) (int, error) {
	if err := validateGeometry(w, h); err != nil {
		return 0, err
	}
	args := s.argv(w, h)
	cmd := exec.CommandContext(ctx, s.cfg.Bin, args...) //nolint:gosec // argv is tokenized from service configuration, never evaluated by a shell.
	stderr := s.captureStderr("ffmpeg", cmd)
	s.configureProc(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("start ffmpeg: %w", err)
	}
	frameSize := w * h * 3
	buf := make([]byte, frameSize)
	delivered := 0
	for {
		if _, err := io.ReadFull(stdout, buf); err != nil {
			waitErr := cmd.Wait()
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return delivered, nil
			}
			if ctx.Err() == nil {
				s.log.Warnf("ffmpeg stream %s ended: %v: %s",
					s.RedactedURL(), waitErr, stderr.firstLine())
			}
			return delivered, err
		}
		f := Frame{Encoding: RGB24, Width: w, Height: h, Bytes: append([]byte(nil), buf...), Time: time.Now()}
		select {
		case out <- f:
			delivered++
		case <-ctx.Done():
			_ = cmd.Wait()
			return delivered, ctx.Err()
		}
	}
}

// inputURL returns the ffmpeg target: the loopback auth-normalizing relay
// when active, otherwise the camera URL. Userinfo is never logged.
func (s *ffmpegSource) inputURL() string {
	if s.normalizer != nil {
		if local, err := s.normalizer.localURL(s.ref.URL); err == nil {
			return local
		}
	}
	return s.ref.URL
}

// argv renders the configured templates into an argv slice. {url} is always
// injected as one standalone argument and no shell is involved.
func (s *ffmpegSource) argv(width, height int) []string {
	args := tokenize(s.cfg.Global)
	key := s.ref.Kind
	tmpl := s.cfg.InputTemplates[key]
	if tmpl == "" {
		tmpl = s.cfg.InputTemplates["rtsp"]
	}
	var inputTokens []string
	filter := fmt.Sprintf("scale=%d:%d", width, height)
	if s.ref.DecodeAccel == DecodeAccelVAAPI {
		device := s.vaapiDevice()
		// Explicit device creation fails at startup when the node or VA
		// driver is missing, unlike a bare -hwaccel vaapi that silently
		// falls back to software. Surfaces stay in VRAM through scaling;
		// hwdownload yields NV12 and the rgb24 output pixel format runs the
		// final cheap CPU conversion.
		args = append(args, "-init_hw_device", "vaapi=va:"+device)
		inputTokens = []string{"-hwaccel", "vaapi", "-hwaccel_device", "va", "-hwaccel_output_format", "vaapi"}
		filter = fmt.Sprintf("scale_vaapi=%d:%d,hwdownload,format=nv12", width, height)
	}
	args = append(args, renderInputTemplate(tmpl, s.inputURL(), s.cfg, inputTokens)...)
	args = append(args, tokenize(s.cfg.ExtraArgs)...)
	args = append(args, tokenize(s.ref.ExtraArgs)...)
	// Guarantee rawvideo output has exactly the ffprobe geometry even when
	// rotation metadata or a mid-stream resolution change would otherwise
	// make stdout reads silently frame data with the wrong stride.
	args = append(args, "-vf", filter)
	args = append(args, tokenize(s.cfg.OutputTemplate)...)
	return args
}

func (s *ffmpegSource) configureProc(cmd *exec.Cmd) {
	attr := &syscall.SysProcAttr{Setpgid: true}
	if os.Geteuid() == 0 && s.cfg.DropUID > 0 && s.cfg.DropGID > 0 {
		uid := uint32(s.cfg.DropUID) //nolint:gosec // DropUID is an operator-configured local unprivileged UID.
		gid := uint32(s.cfg.DropGID) //nolint:gosec // DropGID is an operator-configured local unprivileged GID.
		cred := &syscall.Credential{Uid: uid, Gid: gid, Groups: []uint32{gid}}
		if s.ref.DecodeAccel == DecodeAccelVAAPI {
			// Go clears supplementary groups when Credential is set, so a
			// container-level group_add would not reach the unprivileged
			// child. Grant exactly the render node's owning GID, resolved at
			// runtime so both Docker bind mounts and Kubernetes device-plugin
			// nodes (which carry the host's numeric GID) work.
			if nodeGID, ok := deviceNodeGID(s.vaapiDevice()); ok {
				cred.Groups = append(cred.Groups, nodeGID)
			}
		}
		attr.Credential = cred
	}
	cmd.SysProcAttr = attr
}

// deviceNodeGID returns the owning GID of a device node, used to authorize
// the unprivileged ffmpeg child for the configured VA-API render node.
func deviceNodeGID(path string) (uint32, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Gid == 0 {
		return 0, false
	}
	return stat.Gid, true
}

// renderTemplate substitutes fixed placeholders while preserving the URL as
// one unparsed argv element. This prevents whitespace or token-looking text
// in credentials/query parameters from injecting ffmpeg arguments.
func renderTemplate(tmpl, rawURL string, cfg FFmpegConfig) []string {
	return renderInputTemplate(tmpl, rawURL, cfg, nil)
}

// renderInputTemplate works like renderTemplate but inserts input-side
// decoder options (such as the VA-API hwaccel flags) immediately before the
// input's -i token. Hardware-acceleration flags must precede the input they
// apply to; a template without -i gets them appended at the end of the
// pre-URL token sequence, still an input-side position.
func renderInputTemplate(tmpl, rawURL string, cfg FFmpegConfig, inputTokens []string) []string {
	timeout := strconv.Itoa(maxInt(cfg.ConnectTimeoutSec, 1) * 1_000_000)
	tmpl = strings.ReplaceAll(tmpl, "{timeout_us}", timeout)
	parts := strings.Split(tmpl, "{url}")
	pre := tokenize(parts[0])
	if len(inputTokens) > 0 {
		insertAt := len(pre)
		for i := len(pre) - 1; i >= 0; i-- {
			if pre[i] == "-i" {
				insertAt = i
				break
			}
		}
		inserted := make([]string, 0, len(pre)+len(inputTokens))
		inserted = append(inserted, pre[:insertAt]...)
		inserted = append(inserted, inputTokens...)
		inserted = append(inserted, pre[insertAt:]...)
		pre = inserted
	}
	args := pre
	for _, part := range parts[1:] {
		args = append(args, rawURL)
		args = append(args, tokenize(part)...)
	}
	return args
}

var ffmpegValueFlags = map[string]bool{
	"-r": true, "-ss": true, "-t": true, "-threads": true,
	"-fflags": true, "-flags": true,
}

// ValidateFFmpegExtra validates optional per-job ffmpeg arguments against
// the same allowlist enforced before process execution.
func ValidateFFmpegExtra(extra string) error { return validateExtraArgs(extra) }

func validateExtraArgs(extra string) error {
	tokens := tokenize(extra)
	for i := 0; i < len(tokens); i++ {
		token := tokens[i]
		switch token {
		case "-re":
			continue
		case "-r", "-ss", "-t", "-threads", "-fflags", "-flags":
			if i+1 >= len(tokens) {
				return fmt.Errorf("ffmpeg extra option %s requires a value", token)
			}
			i++
		default:
			if strings.HasPrefix(token, "-") && ffmpegValueFlags[strings.SplitN(token, "=", 2)[0]] {
				continue
			}
			return fmt.Errorf("ffmpeg extra argument %q is not in the allowlist", token)
		}
	}
	return nil
}

// tokenize splits on whitespace; templates intentionally do not support
// quoted arguments because every value arrives as a single argv element.
func tokenize(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return strings.Fields(s)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// embeddedCredentialURL matches rtsp/http(s) URLs carrying userinfo inside a
// free-form log line.
var embeddedCredentialURL = regexp.MustCompile(`(?i)\b((?:rtsp|https?)://)[^\s/@"'<>]+@`)

// redactText strips userinfo from any URL embedded in an ffmpeg stderr line.
func redactText(line string) string {
	return embeddedCredentialURL.ReplaceAllString(line, "${1}***@")
}

// stderrCapture drains one child-process stderr stream: every redacted line
// goes to the service logger at DEBUG, and the first line is retained so a
// failed start can summarize the cause at WARN.
type stderrCapture struct {
	phase string
	first string
}

func (s *ffmpegSource) captureStderr(phase string, cmd *exec.Cmd) *stderrCapture {
	cap := &stderrCapture{phase: phase}
	r, err := cmd.StderrPipe()
	if err != nil {
		cap.first = "stderr unavailable"
		return cap
	}
	go func() {
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 4096), 256*1024)
		for scanner.Scan() {
			line := redactText(scanner.Text())
			if cap.first == "" {
				cap.first = line
			}
			s.log.Debugf("ffmpeg %s [%s]: %s", phase, s.RedactedURL(), line)
		}
	}()
	return cap
}

func (c *stderrCapture) firstLine() string {
	if c == nil || c.first == "" {
		return "no stderr output"
	}
	return c.first
}

func nextBackoff(cur, max time.Duration) time.Duration {
	cur *= 2
	if cur > max {
		return max
	}
	return cur
}

// sleepCtx waits for d or ctx cancellation; true means canceled.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return true
	case <-t.C:
		return false
	}
}
