// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"bufio"
	"bytes"
	"io"
	"net"
	neturl "net/url"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNormalizeAuthChallenge(t *testing.T) {
	triple := []byte("RTSP/1.0 401 Unauthorized\r\n" +
		"CSeq: 2\r\n" +
		`WWW-Authenticate: Digest realm="Camera", nonce="abc"` + "\r\n" +
		`WWW-Authenticate: Digest realm="Camera", nonce="abc", algorithm=SHA-256` + "\r\n" +
		`WWW-Authenticate: Basic realm="Camera"` + "\r\n" +
		"\r\n")
	out := normalizeAuthChallenge(triple, nopLogger{})
	if !rtspStatus401.Match(out) {
		t.Fatalf("status line altered: %q", out)
	}
	if bytes.Contains(out, []byte("Digest")) {
		t.Fatalf("Digest challenge not stripped: %q", out)
	}
	if !bytes.Contains(out, []byte("Basic realm")) {
		t.Fatalf("Basic challenge missing: %q", out)
	}
	if !bytes.HasSuffix(out, []byte("\r\n\r\n")) {
		t.Fatalf("header block must end with a single blank line: %q", out)
	}
	if c := bytes.Count(out, []byte("CSeq")); c != 1 {
		t.Fatalf("CSeq count = %d, want 1", c)
	}

	// Digest-only cameras must be left untouched.
	digestOnly := bytes.ReplaceAll(triple, []byte(`WWW-Authenticate: Basic realm="Camera"`+"\r\n"), nil)
	if got := normalizeAuthChallenge(digestOnly, nopLogger{}); !bytes.Equal(got, digestOnly) {
		t.Fatalf("Digest-only challenge was modified:\n%s", got)
	}

	// Non-401 responses never change.
	ok := []byte("RTSP/1.0 200 OK\r\nContent-Length: 4\r\n\r\nabcd")
	if got := normalizeAuthChallenge(ok, nopLogger{}); !bytes.Equal(got, ok) {
		t.Fatalf("200 response was modified: %q", got)
	}
}

func TestRTSPAuthNormalizerRelay(t *testing.T) {
	challenge := []byte("RTSP/1.0 401 Unauthorized\r\n" +
		"CSeq: 1\r\n" +
		`WWW-Authenticate: Digest realm="Camera", nonce="abc"` + "\r\n" +
		`WWW-Authenticate: Basic realm="Camera"` + "\r\n" +
		"\r\n")
	body := []byte("v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=stream\r\n")
	// '$' + channel 0 + 2-byte big-endian length 5 + payload.
	interleaved := []byte{'$', 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o'}

	camera, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer camera.Close()

	go func() {
		up, err := camera.Accept()
		if err != nil {
			return
		}
		defer up.Close()
		reader := bufio.NewReader(up)
		readRequest := func() {
			for {
				line, err := reader.ReadBytes('\n')
				if err != nil {
					return
				}
				if strings.TrimRight(string(line), "\r\n") == "" {
					return
				}
			}
		}
		// First request: multi-scheme 401.
		readRequest()
		_, _ = up.Write(challenge)
		// Retried request: 200 with SDP body, then one interleaved frame.
		readRequest()
		resp := "RTSP/1.0 200 OK\r\n" +
			"CSeq: 2\r\n" +
			"Content-Type: application/sdp\r\n" +
			"Content-Length: " + strconv.Itoa(len(body)) + "\r\n" +
			"\r\n"
		_, _ = up.Write([]byte(resp))
		_, _ = up.Write(body)
		_, _ = up.Write(interleaved)
		// Keep the connection open briefly so the client can drain.
		time.Sleep(300 * time.Millisecond)
	}()

	rawURL := "rtsp://" + camera.Addr().String() + "/Streaming/Channels/101"
	n, err := startRTSPAuthNormalizer(rawURL, nopLogger{})
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	local, err := n.localURL(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(local, "rtsp://127.0.0.1:") {
		t.Fatalf("local URL = %q, want loopback relay", local)
	}
	if !strings.HasSuffix(local, "/Streaming/Channels/101") {
		t.Fatalf("path not preserved: %q", local)
	}

	parsedLocal, err := neturl.Parse(local)
	if err != nil {
		t.Fatal(err)
	}
	conn, err := net.DialTimeout("tcp", parsedLocal.Host, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(5 * time.Second))

	if _, err := conn.Write([]byte("DESCRIBE " + rawURL + " RTSP/1.0\r\nCSeq: 1\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	block1 := readHeaderBlock(t, conn)
	if bytes.Contains(block1, []byte("Digest")) || !bytes.Contains(block1, []byte("Basic")) {
		t.Fatalf("challenge not normalized: %q", block1)
	}

	if _, err := conn.Write([]byte("DESCRIBE " + rawURL + " RTSP/1.0\r\nCSeq: 2\r\nAuthorization: Basic xxx\r\n\r\n")); err != nil {
		t.Fatal(err)
	}
	block2 := readHeaderBlock(t, conn)
	m := regexp.MustCompile(`(?im)^Content-Length:\s*(\d+)\s*$`).FindSubmatch(block2)
	if m == nil {
		t.Fatalf("no Content-Length in %q", block2)
	}
	nBody, _ := strconv.Atoi(string(m[1]))
	gotBody := make([]byte, nBody)
	if _, err := io.ReadFull(conn, gotBody); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotBody, body) {
		t.Fatalf("SDP body mismatch: %q", gotBody)
	}
	gotFrame := make([]byte, len(interleaved))
	if _, err := io.ReadFull(conn, gotFrame); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotFrame, interleaved) {
		t.Fatalf("interleaved frame framing broken: %v", gotFrame)
	}
}

func readHeaderBlock(t *testing.T, r io.Reader) []byte {
	t.Helper()
	var buf bytes.Buffer
	one := make([]byte, 1)
	for {
		if _, err := io.ReadFull(r, one); err != nil {
			t.Fatal(err)
		}
		buf.WriteByte(one[0])
		if bytes.HasSuffix(buf.Bytes(), []byte("\r\n\r\n")) {
			return buf.Bytes()
		}
		if buf.Len() > maxRTSPHeaderBytes {
			t.Fatal("header block too large")
		}
	}
}

func TestNormalizeStreamURI(t *testing.T) {
	cases := []struct {
		name    string
		cfg     FederationConfig
		in      string
		want    string
		changed bool
	}{
		{
			name:    "aggregate-channel-to-main",
			cfg:     FederationConfig{NormalizeVirtualChannel: true, MainChannel: "101"},
			in:      "rtsp://camera:554/Streaming/Channels/0?transportmode=unicast&profile=Profile_1",
			want:    "rtsp://camera:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1",
			changed: true,
		},
		{
			name: "case-insensitive-prefix",
			cfg:  FederationConfig{NormalizeVirtualChannel: true, MainChannel: "101"},
			in:   "rtsp://camera:554/streaming/channels/0",
			want: "rtsp://camera:554/streaming/channels/101",
		},
		{
			name: "playable-channel-untouched",
			cfg:  FederationConfig{NormalizeVirtualChannel: true, MainChannel: "101"},
			in:   "rtsp://camera:554/Streaming/Channels/101",
			want: "rtsp://camera:554/Streaming/Channels/101",
		},
		{
			name: "other-path-untouched",
			cfg:  FederationConfig{NormalizeVirtualChannel: true, MainChannel: "101"},
			in:   "rtsp://camera:554/live.mp4",
			want: "rtsp://camera:554/live.mp4",
		},
		{
			name: "flag-off",
			cfg:  FederationConfig{NormalizeVirtualChannel: false, MainChannel: "101"},
			in:   "rtsp://camera:554/Streaming/Channels/0",
			want: "rtsp://camera:554/Streaming/Channels/0",
		},
		{
			name: "custom-main-channel",
			cfg:  FederationConfig{NormalizeVirtualChannel: true, MainChannel: "201"},
			in:   "rtsp://camera:554/Streaming/Channels/0",
			want: "rtsp://camera:554/Streaming/Channels/201",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := NewFederation("http://metadata", "http://command", nil, tc.cfg)
			if got := f.normalizeStreamURI(tc.in); got != tc.want {
				t.Fatalf("normalizeStreamURI() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestRedactText(t *testing.T) {
	line := "Authorization: failing open rtsp://admin:placeholder-pw@192.0.2.10:554/Streaming/Channels/101: 401 Unauthorized"
	got := redactText(line)
	if strings.Contains(got, "placeholder-pw") {
		t.Fatalf("password leaked into log: %q", got)
	}
	if !strings.Contains(got, "rtsp://***@192.0.2.10") {
		t.Fatalf("expected redacted URL, got %q", got)
	}
	if redactText("plain ffmpeg note without url") != "plain ffmpeg note without url" {
		t.Fatal("plain text must not be modified")
	}
}
