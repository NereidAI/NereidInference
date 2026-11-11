// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"bufio"
	"bytes"
	"io"
	"net"
	"net/url"
	"regexp"
	"strconv"
	"sync"
	"time"
)

// Some ONVIF cameras answer RTSP DESCRIBE with several WWW-Authenticate
// challenges at once (Digest MD5, Digest SHA-256 and Basic). libavformat in
// ffmpeg 6.x fails to answer such a multi-scheme response: its retried
// request carries no Authorization header at all and the handshake loops on
// 401, even though the camera accepts a manually built Basic (or single
// Digest) response.
//
// rtspAuthNormalizer is a loopback-only byte relay between ffmpeg and the
// camera that rewrites exactly one thing: a 401 response offering both Basic
// and Digest has its Digest challenges removed, leaving the single Basic
// challenge libavformat handles correctly. When a 401 offers only Digest
// challenges the response is forwarded untouched. Interleaved (RTP-over-TCP)
// frames are forwarded byte for byte; credentials never appear in logs.
//
// It applies to the TCP rtsp source kind only. rtsp-udp media would not
// traverse the relay and is therefore never proxied.
type rtspAuthNormalizer struct {
	listen net.Listener
	target string // camera host:port
	log    Logger

	conns   sync.WaitGroup
	closeMu sync.Mutex
	closed  bool
}

// startRTSPAuthNormalizer binds a loopback listener that relays to the RTSP
// authority of rawURL. Call Close to release it.
func startRTSPAuthNormalizer(rawURL string, log Logger) (*rtspAuthNormalizer, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}
	port := u.Port()
	if port == "" {
		port = "554"
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	n := &rtspAuthNormalizer{
		listen: ln,
		target: net.JoinHostPort(u.Hostname(), port),
		log:    log,
	}
	go n.serve()
	return n, nil
}

// localURL rewrites the camera URL to the loopback relay while preserving
// userinfo, path and query.
func (n *rtspAuthNormalizer) localURL(rawURL string) (string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	u.Host = n.listen.Addr().String()
	return u.String(), nil
}

func (n *rtspAuthNormalizer) Close() error {
	n.closeMu.Lock()
	n.closed = true
	n.closeMu.Unlock()
	err := n.listen.Close()
	n.conns.Wait()
	return err
}

func (n *rtspAuthNormalizer) serve() {
	for {
		conn, err := n.listen.Accept()
		if err != nil {
			return
		}
		n.conns.Add(1)
		go n.handle(conn)
	}
}

func (n *rtspAuthNormalizer) handle(client net.Conn) {
	defer n.conns.Done()
	upstream, err := net.DialTimeout("tcp", n.target, 10*time.Second)
	if err != nil {
		n.log.Warnf("rtsp auth normalizer dial %s failed: %v", n.target, err)
		_ = client.Close()
		return
	}

	var once sync.Once
	closeBoth := func() {
		_ = client.Close()
		_ = upstream.Close()
	}
	defer once.Do(closeBoth)

	var pump sync.WaitGroup
	pump.Add(2)
	// Client -> camera: pure byte relay (RTSP requests are always text;
	// interleaved binary frames only flow server -> client).
	go func() {
		defer pump.Done()
		_, _ = io.Copy(upstream, client)
		once.Do(closeBoth)
	}()
	// Camera -> client: framing-aware relay that can rewrite a 401 block.
	go func() {
		defer pump.Done()
		n.relayResponses(client, bufio.NewReaderSize(upstream, 64*1024))
		once.Do(closeBoth)
	}()
	pump.Wait()
}

// relayResponses forwards RTSP responses and interleaved frames. It parses
// header blocks to reach the Content-Length body boundary so mixed
// text/binary traffic after PLAY keeps exact framing.
func (n *rtspAuthNormalizer) relayResponses(dst io.Writer, src *bufio.Reader) {
	for {
		first, err := src.Peek(1)
		if err != nil {
			return
		}
		if first[0] == interleavedFrameMarker {
			if !forwardInterleaved(dst, src) {
				return
			}
			continue
		}
		var header bytes.Buffer
		for {
			line, err := src.ReadBytes('\n')
			if err != nil {
				return
			}
			header.Write(line)
			if bytes.Equal(bytes.TrimRight(line, "\r\n"), []byte("")) {
				break
			}
			if header.Len() > maxRTSPHeaderBytes {
				return
			}
		}
		block := header.Bytes()
		bodyLen := rtspContentLength(block)
		block = normalizeAuthChallenge(block, n.log)
		if _, err := dst.Write(block); err != nil {
			return
		}
		if bodyLen > 0 {
			if _, err := io.CopyN(dst, src, int64(bodyLen)); err != nil {
				return
			}
		}
	}
}

func forwardInterleaved(dst io.Writer, src *bufio.Reader) bool {
	// '$' + 1-byte channel + 2-byte big-endian length + payload.
	head := make([]byte, 4)
	if _, err := io.ReadFull(src, head); err != nil {
		return false
	}
	if _, err := dst.Write(head); err != nil {
		return false
	}
	n := int(head[2])<<8 | int(head[3])
	if n > maxInterleavedFrameBytes {
		return false
	}
	if n > 0 {
		if _, err := io.CopyN(dst, src, int64(n)); err != nil {
			return false
		}
	}
	return true
}

const (
	interleavedFrameMarker   = '$'
	maxRTSPHeaderBytes       = 64 * 1024
	maxInterleavedFrameBytes = 8 * 1024 * 1024
)

var (
	rtspStatus401        = regexp.MustCompile(`^RTSP/\d(?:\.\d)? 401\b`)
	contentLengthPattern = regexp.MustCompile(`(?im)^Content-Length:\s*(\d+)\s*$`)
	wwwAuthPattern       = regexp.MustCompile(`(?i)^WWW-Authenticate:\s*(\w+)`)
)

func rtspContentLength(block []byte) int {
	m := contentLengthPattern.FindSubmatch(block)
	if m == nil {
		return 0
	}
	n, err := strconv.Atoi(string(m[1]))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// normalizeAuthChallenge strips Digest challenges from a 401 that also
// offers Basic. Responses without a Basic alternative pass untouched so
// Digest-only cameras keep working.
func normalizeAuthChallenge(block []byte, log Logger) []byte {
	if !rtspStatus401.Match(block) {
		return block
	}
	hasBasic, hasDigest := false, false
	for _, rawLine := range bytes.Split(block, []byte("\n")) {
		m := wwwAuthPattern.FindSubmatch(bytes.TrimRight(rawLine, "\r"))
		if m == nil {
			continue
		}
		switch string(bytes.ToLower(m[1])) {
		case "basic":
			hasBasic = true
		case "digest":
			hasDigest = true
		}
	}
	if !hasBasic || !hasDigest {
		return block
	}
	var out bytes.Buffer
	for _, rawLine := range bytes.Split(block, []byte("\n")) {
		line := bytes.TrimRight(rawLine, "\r")
		if len(line) == 0 {
			// Header/body separator; re-added exactly once below.
			continue
		}
		if m := wwwAuthPattern.FindSubmatch(line); m != nil &&
			string(bytes.ToLower(m[1])) == "digest" {
			continue
		}
		out.Write(line)
		out.WriteString("\r\n")
	}
	out.WriteString("\r\n")
	log.Debugf("rtsp 401 offered Basic+Digest; normalized to Basic-only challenge")
	return out.Bytes()
}
