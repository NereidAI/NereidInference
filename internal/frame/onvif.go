// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package frame

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

const maxONVIFJSON = 4 * 1024 * 1024

// SecretGetter returns the key/value pairs of a named EdgeX secret.
type SecretGetter func(secretName string) (map[string]string, error)

// FederationConfig parameterizes ONVIF federation.
type FederationConfig struct {
	// NormalizeVirtualChannel rewrites the non-playable aggregate RTSP
	// channel ".../Streaming/Channels/0" returned by some firmware to the
	// main profile channel (MainChannel, default "101"). The rewrite stays
	// generic: it matches the ONVIF/RTSP virtual-channel convention rather
	// than a vendor name and can be disabled for conformant cameras.
	NormalizeVirtualChannel bool
	// MainChannel is the channel segment used by the virtual-channel
	// rewrite; "101" is the conventional main-stream channel.
	MainChannel string
	// Log receives structured diagnostics; nil selects a no-op logger.
	Log Logger
}

// Federation resolves camera EdgeX device names into live RTSP URLs using
// data already registered by device-onvif-camera: protocol properties from
// core-metadata, the GetStreamUri command from core-command, and credentials
// from the EdgeX secret store. The DS never asks NereidEdge or an operator to
// type a URL, and it never logs embedded credentials.
type Federation struct {
	metadataURL string
	commandURL  string
	secrets     SecretGetter
	http        *http.Client

	normalizeVirtualChannel bool
	mainChannel             string
	log                     Logger

	mu    sync.Mutex
	cache map[cachedStreamKey]cachedURI
}

// cachedStreamKey scopes a resolved URI to one camera AND one ONVIF media
// profile token: main and sub streams of the same camera are distinct RTSP
// URIs and must never share a cache entry. The empty token is the device
// service default profile.
type cachedStreamKey struct {
	device  string
	profile string
}

type cachedURI struct {
	ref ResolvedRef
	at  time.Time
}

// NewFederation builds a federation resolver.
func NewFederation(metadataURL, commandURL string, secrets SecretGetter, cfg FederationConfig) *Federation {
	mainChannel := strings.TrimSpace(cfg.MainChannel)
	if mainChannel == "" {
		mainChannel = defaultMainChannel
	}
	return &Federation{
		metadataURL:             strings.TrimRight(metadataURL, "/"),
		commandURL:              strings.TrimRight(commandURL, "/"),
		secrets:                 secrets,
		http:                    &http.Client{Timeout: 15 * time.Second},
		normalizeVirtualChannel: cfg.NormalizeVirtualChannel,
		mainChannel:             mainChannel,
		log:                     loggerOrNop(cfg.Log),
		cache:                   map[cachedStreamKey]cachedURI{},
	}
}

const defaultMainChannel = "101"

// Resolve returns an RTSP source reference for an EdgeX camera device name.
// secretHint names a secret already available to this device service; the
// camera's Protocols.Onvif.SecretName value is used when the caller does not
// provide one. profileToken selects a specific ONVIF media profile
// (main/sub/third stream); an empty token lets the camera service pick its
// default profile. Different profile tokens resolve independently and are
// cached independently.
func (f *Federation) Resolve(ctx context.Context, deviceName, secretHint, profileToken string) (ResolvedRef, error) {
	profileToken = strings.TrimSpace(profileToken)
	key := cachedStreamKey{device: deviceName, profile: profileToken}
	if r, ok := f.cached(key); ok {
		f.log.Debugf("onvif federation cache hit for camera %s profile %q -> %s", deviceName, profileToken, RedactUserInfo(r.URL))
		return r, nil
	}
	props, metadataSecret, err := f.fetchDevice(ctx, deviceName)
	if err != nil {
		f.log.Warnf("onvif federation metadata lookup for %s failed: %v", deviceName, err)
		return ResolvedRef{}, err
	}
	secretName := firstNonEmpty(secretHint, metadataSecret)
	streamURL, err := f.fetchStreamURI(ctx, deviceName, profileToken)
	if err != nil {
		f.log.Warnf("onvif federation stream URI for %s failed: %v", deviceName, err)
		return ResolvedRef{}, err
	}
	if normalized := f.normalizeStreamURI(streamURL); normalized != streamURL {
		f.log.Infof("onvif federation normalized non-playable stream URI for %s: %s -> %s",
			deviceName, RedactUserInfo(streamURL), RedactUserInfo(normalized))
		streamURL = normalized
	}
	streamURL, err = injectCredentials(streamURL, secretName, f.secrets)
	if err != nil {
		f.log.Warnf("onvif federation credential injection for %s failed: %v", deviceName, err)
		return ResolvedRef{}, err
	}
	ref := ResolvedRef{Kind: "rtsp", URL: streamURL}
	f.log.Debugf("onvif federation resolved camera %s profile %q -> %s", deviceName, profileToken, RedactUserInfo(streamURL))
	f.mu.Lock()
	f.cache[key] = cachedURI{ref: ref, at: time.Now()}
	f.mu.Unlock()
	_ = props
	return ref, nil
}

// Invalidate drops cached URIs for every profile of a camera (called on 401
// or stream setup failure).
func (f *Federation) Invalidate(deviceName string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for key := range f.cache {
		if key.device == deviceName {
			delete(f.cache, key)
		}
	}
}

func (f *Federation) cached(key cachedStreamKey) (ResolvedRef, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	c, ok := f.cache[key]
	if ok && time.Since(c.at) < 5*time.Minute {
		return c.ref, true
	}
	return ResolvedRef{}, false
}

// metadataDevice is the tolerant view of an EdgeX v4 device payload.
type metadataDevice struct {
	Device struct {
		Protocols map[string]map[string]any `json:"protocols"`
	} `json:"device"`
}

func (f *Federation) fetchDevice(ctx context.Context, name string) (map[string]any, string, error) {
	u := fmt.Sprintf("%s/api/v3/device/name/%s", f.metadataURL, url.PathEscape(name))
	body, _, err := f.getJSON(ctx, u)
	if err != nil {
		return nil, "", fmt.Errorf("metadata lookup camera %s: %w", name, err)
	}
	var d metadataDevice
	if err := json.Unmarshal(body, &d); err != nil {
		return nil, "", err
	}
	for protoName, props := range d.Device.Protocols {
		if !strings.EqualFold(protoName, "Onvif") {
			continue
		}
		return props, stringProp(props, "SecretName"), nil
	}
	return nil, "", fmt.Errorf("camera %s has no Onvif protocol properties", name)
}

// commandResponse is the tolerant view of a v4 command event response.
type commandResponse struct {
	Event struct {
		Readings []struct {
			ResourceName string `json:"resourceName"`
			ValueType    string `json:"valueType"`
			Value        string `json:"value"`
			ObjectValue  any    `json:"objectValue"`
			BinaryValue  []byte `json:"binaryValue"`
		} `json:"readings"`
	} `json:"event"`
}

func (f *Federation) fetchStreamURI(ctx context.Context, name, profileToken string) (string, error) {
	// A specific ONVIF media profile selects a particular encoded stream
	// (main/sub/third). On device-onvif-camera 4.x the StreamUri device
	// resource is read-only: the profile token travels in a GET query
	// parameter "jsonObject" whose value is the base64 encoding of
	// {"ProfileToken": ..., "StreamSetup": {Stream, Transport}}. PUT is
	// rejected 405 and arbitrary query keys are ignored.
	if profileToken != "" {
		query, err := streamURIQuery(profileToken)
		if err != nil {
			return "", err
		}
		// 4.0.2 registers only StreamUri; older releases spell the command
		// GetStreamUri. Both accept the same jsonObject query.
		for _, cmd := range []string{"StreamUri", "GetStreamUri"} {
			u := fmt.Sprintf("%s/api/v3/device/name/%s/%s?jsonObject=%s",
				f.commandURL, url.PathEscape(name), cmd, query)
			body, status, err := f.getJSON(ctx, u)
			if err != nil {
				f.log.Debugf("onvif federation %s profile %q for camera %s failed (http=%d): %v",
					cmd, profileToken, name, status, err)
				continue
			}
			if uri := parseStreamURI(body); uri != "" {
				f.log.Debugf("onvif federation %s profile %q for camera %s returned %s",
					cmd, profileToken, name, RedactUserInfo(uri))
				return uri, nil
			}
			f.log.Debugf("onvif federation %s profile %q for camera %s returned no usable rtsp URL",
				cmd, profileToken, name)
		}
		return "", fmt.Errorf("no profile-scoped stream URI command for camera %s profile %q", name, profileToken)
	}
	// With no token the default-profile URI needs no jsonObject; the camera
	// service applies profile/stream defaults itself.
	for _, cmd := range []string{"StreamUri", "GetStreamUri"} {
		u := fmt.Sprintf("%s/api/v3/device/name/%s/%s", f.commandURL, url.PathEscape(name), cmd)
		body, status, err := f.getJSON(ctx, u)
		if err != nil {
			f.log.Debugf("onvif federation %s for camera %s failed (http=%d): %v", cmd, name, status, err)
			continue
		}
		if uri := parseStreamURI(body); uri != "" {
			f.log.Debugf("onvif federation %s for camera %s returned %s", cmd, name, RedactUserInfo(uri))
			return uri, nil
		}
		f.log.Debugf("onvif federation %s for camera %s returned no usable rtsp URL", cmd, name)
	}
	return "", fmt.Errorf("no usable stream URI command for camera %s", name)
}

// streamSetupRequest is the JSON object device-onvif-camera expects inside
// the StreamUri command's jsonObject query parameter.
type streamSetupRequest struct {
	ProfileToken string      `json:"ProfileToken"`
	StreamSetup  streamSetup `json:"StreamSetup"`
}

type streamSetup struct {
	Stream    string          `json:"Stream"`
	Transport streamTransport `json:"Transport"`
}

type streamTransport struct {
	Protocol string `json:"Protocol"`
}

// streamURIQuery returns the URL-escaped base64 jsonObject value selecting an
// RTP-over-RTSP/TCP unicast stream for profileToken.
func streamURIQuery(profileToken string) (string, error) {
	raw, err := json.Marshal(streamSetupRequest{
		ProfileToken: profileToken,
		StreamSetup: streamSetup{
			Stream:    "RTP-Unicast",
			Transport: streamTransport{Protocol: "RTSP"},
		},
	})
	if err != nil {
		return "", err
	}
	return url.QueryEscape(base64.StdEncoding.EncodeToString(raw)), nil
}

// virtualChannelPath matches the aggregate RTSP channel 0 some ONVIF
// firmware returns from GetStreamUri; RTSP rejects it with 400 and the
// conventional main channel (".../Streaming/Channels/101") must be used.
var virtualChannelPath = regexp.MustCompile(`(?i)^(.*?/streaming/channels/)0$`)

// normalizeStreamURI applies the configured firmware-compat rewrite. It is
// deliberately path-based and off by a flag, not a vendor special case.
func (f *Federation) normalizeStreamURI(rawURL string) string {
	if !f.normalizeVirtualChannel {
		return rawURL
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return rawURL
	}
	m := virtualChannelPath.FindStringSubmatch(parsed.Path)
	if m == nil {
		return rawURL
	}
	parsed.Path = m[1] + f.mainChannel
	return parsed.String()
}

func (f *Federation) getJSON(ctx context.Context, u string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := f.http.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode, fmt.Errorf("HTTP %d from %s", resp.StatusCode, RedactUserInfo(u))
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxONVIFJSON+1))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if int64(len(b)) > maxONVIFJSON {
		return nil, resp.StatusCode, fmt.Errorf("ONVIF response from %s exceeds %d bytes", RedactUserInfo(u), maxONVIFJSON)
	}
	return b, resp.StatusCode, nil
}

// parseStreamURI extracts an rtsp:// URL from a command response, tolerating
// both plain string values and JSON-encoded objects.
func parseStreamURI(body []byte) string {
	var cr commandResponse
	if err := json.Unmarshal(body, &cr); err != nil {
		return ""
	}
	for _, r := range cr.Event.Readings {
		if uri := rtspFromString(r.Value); uri != "" {
			return uri
		}
		if uri := rtspFromObject(r.ObjectValue); uri != "" {
			return uri
		}
	}
	return ""
}

func isRTSPURL(s string) bool {
	return len(s) >= len("rtsp://") && strings.EqualFold(s[:len("rtsp://")], "rtsp://")
}

func rtspFromString(s string) string {
	if isRTSPURL(s) {
		return s
	}
	var nested map[string]any
	if err := json.Unmarshal([]byte(s), &nested); err != nil {
		return ""
	}
	return rtspFromObject(nested)
}

func rtspFromObject(v any) string {
	switch t := v.(type) {
	case string:
		return rtspFromString(t)
	case map[string]any:
		return rtspFromMap(t)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		var nested map[string]any
		if err := json.Unmarshal(b, &nested); err != nil {
			return ""
		}
		return rtspFromMap(nested)
	}
}

func rtspFromMap(m map[string]any) string {
	for _, key := range []string{"uri", "Uri", "URI", "streamUri", "StreamUri", "url", "URL"} {
		if v := stringProp(m, key); isRTSPURL(v) {
			return v
		}
	}
	for _, v := range m {
		if uri := rtspFromObject(v); uri != "" {
			return uri
		}
	}
	return ""
}

// InjectCredentials adds percent-encoded userinfo from the EdgeX secret
// store to a URL. URLs that already carry userinfo are returned unchanged.
func InjectCredentials(rawURL, secretName string, get SecretGetter) (string, error) {
	return injectCredentials(rawURL, secretName, get)
}

// injectCredentials adds percent-encoded userinfo to a URL.
func injectCredentials(rawURL, secretName string, get SecretGetter) (string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	if u.User != nil {
		return rawURL, nil
	}
	if secretName == "" || get == nil {
		return rawURL, nil
	}
	secret, err := get(secretName)
	if err != nil {
		return "", fmt.Errorf("resolve camera secret %s: %w", secretName, err)
	}
	user := firstNonEmpty(secret["username"], secret["Username"], secret["user"])
	pass := firstNonEmpty(secret["password"], secret["Password"], secret["pass"])
	if user == "" {
		return rawURL, nil
	}
	u.User = url.UserPassword(user, pass)
	return u.String(), nil
}

func stringProp(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
