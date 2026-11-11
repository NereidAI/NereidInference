// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package config holds the custom "Inference" configuration section of the
// device service. It is loaded by the SDK config processor under that
// section and hot-reloaded through UpdateFromRaw.
package config

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// channelTokenPattern bounds the RTSP channel segment used by the ONVIF
// virtual-channel rewrite (e.g. "101"); it is appended to a URL path and must
// never carry path separators or shell-significant text.
var channelTokenPattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// ServiceConfig wraps the top-level Inference section in configuration.yaml.
// EdgeX custom configuration must be wrapped by a field whose name matches the
// section passed to DeviceServiceSDK.LoadCustomConfig.
type ServiceConfig struct {
	Inference InferenceConfig `json:"Inference" yaml:"Inference"`
}

// UpdateFromRaw replaces configuration from the form emitted by Keeper.
func (c *ServiceConfig) UpdateFromRaw(rawConfig interface{}) bool {
	next, ok := rawConfig.(*ServiceConfig)
	if !ok {
		return false
	}
	*c = *next
	return true
}

// InferenceConfig is the root custom configuration section.
type InferenceConfig struct {
	// DefaultEndpoint configures the node-local managed OVMS KServe gRPC
	// endpoint (host:port) carried by its leases. An explicit job Endpoint is
	// an intentional override; unmanaged backends may also use this default.
	DefaultEndpoint string `json:"DefaultEndpoint" yaml:"DefaultEndpoint"`
	// OpenAIEndpoint is the default base URL for openai-backend models.
	OpenAIEndpoint string `json:"OpenAIEndpoint" yaml:"OpenAIEndpoint"`
	// OpenAIAPISecret names the EdgeX secret entry holding the bearer token
	// ("api_key" key); empty means no authentication.
	OpenAIAPISecret string          `json:"OpenAIAPISecret" yaml:"OpenAIAPISecret"`
	ModelRoot       string          `json:"ModelRoot" yaml:"ModelRoot"`
	Ess             EssConfig       `yaml:"Ess"`
	Inbox           InboxConfig     `yaml:"Inbox"`
	FFmpeg          FFmpegConfig    `yaml:"FFmpeg"`
	Onvif           OnvifConfig     `yaml:"Onvif"`
	Preview         PreviewConfig   `yaml:"Preview"`
	OVMS            OVMSConfig      `yaml:"OVMS"`
	Triton          TritonConfig    `yaml:"Triton"`
	External        ExternalConfig  `yaml:"External"`
	Pipelines       PipelinesConfig `yaml:"Pipelines"`
	// CoreMetadataURL / CoreCommandURL are used in non-secure mode for the
	// ONVIF stream-URI federation. Defaults target the standard EdgeX host.
	CoreMetadataURL string       `json:"CoreMetadataURL" yaml:"CoreMetadataURL"`
	CoreCommandURL  string       `json:"CoreCommandURL" yaml:"CoreCommandURL"`
	Limits          LimitsConfig `yaml:"Limits"`
}

// EssConfig configures the Open Horizon ESS local-object watcher.
type EssConfig struct {
	Enabled  bool   `yaml:"Enabled"`
	LocalDir string `yaml:"LocalDir"`
}

// InboxConfig configures the operator tarball drop directory.
type InboxConfig struct {
	Enabled bool   `yaml:"Enabled"`
	Dir     string `yaml:"Dir"`
}

// FFmpegConfig parameterizes the child-process frame source; the service
// never embeds a parameter table in code.
type FFmpegConfig struct {
	Bin                    string            `yaml:"Bin"`
	ProbeBin               string            `yaml:"ProbeBin"`
	Global                 string            `yaml:"Global"`
	ConnectTimeoutSec      int               `yaml:"ConnectTimeoutSec"`
	InputTemplates         map[string]string `yaml:"InputTemplates"`
	OutputTemplate         string            `yaml:"OutputTemplate"`
	ExtraArgs              string            `yaml:"ExtraArgs"`
	AuthChallengeNormalize bool              `yaml:"AuthChallengeNormalize"`
	// DropUID/DropGID run the ffmpeg children unprivileged when the service
	// itself runs as root (ESS object reads require root under OH).
	DropUID int `yaml:"DropUID"`
	DropGID int `yaml:"DropGID"`
	// VAAPIDevice is the DRM render node opened for jobs whose
	// DecodeAccel=vaapi; the node must be mapped into the container.
	VAAPIDevice string `yaml:"VAAPIDevice"`
}

// OnvifConfig parameterizes ONVIF federation (device-onvif-camera
// core-metadata/core-command resolution of RTSP URLs).
type OnvifConfig struct {
	// NormalizeVirtualChannel rewrites the aggregate ".../Channels/0"
	// URI some firmware returns to the playable main channel.
	NormalizeVirtualChannel bool   `yaml:"NormalizeVirtualChannel"`
	MainChannel             string `yaml:"MainChannel"`
}

// PreviewConfig configures the local annotated MJPEG preview server.
type PreviewConfig struct {
	Enabled bool   `yaml:"Enabled"`
	Bind    string `yaml:"Bind"`
}

// OVMSConfig configures DS-owned multi-model OVMS configuration reconciliation.
type OVMSConfig struct {
	Enabled          bool   `yaml:"Enabled"`
	RESTEndpoint     string `yaml:"RESTEndpoint"`
	ContainerRoot    string `yaml:"ContainerRoot"`
	ReloadTimeoutSec int    `yaml:"ReloadTimeoutSec"`
}

// TritonConfig configures the managed Triton reconciler: NI renders one
// explicit-model-control repository under ModelRoot and drives the HTTP
// Repository API while jobs call the OIP gRPC endpoint through leases.
type TritonConfig struct {
	Enabled         bool   `yaml:"Enabled"`
	ControlEndpoint string `yaml:"ControlEndpoint"`
	GRPCEndpoint    string `yaml:"GRPCEndpoint"`
	ContainerRoot   string `yaml:"ContainerRoot"`
	LoadTimeoutSec  int    `yaml:"LoadTimeoutSec"`
	ReadyTimeoutSec int    `yaml:"ReadyTimeoutSec"`
}

// ExternalConfig configures the read-only health registry for inference
// servers NI cannot administrate. ProvisionFile is deployment material; legacy
// runtime records under ModelRoot remain visible, but managed PUT is rejected.
type ExternalConfig struct {
	Enabled       bool   `yaml:"Enabled"`
	ProvisionFile string `yaml:"ProvisionFile"`
}

// PipelinesConfig selects the opt-in provisioned NI-native graph table. The
// platform ships no business pipeline, so the default is empty: applications
// supply graphs through this read-only YAML file (deployment material copied
// from samples/) or through runtime PipelinesJSON PUTs, which always persist
// below ModelRoot independently of this setting.
type PipelinesConfig struct {
	ProvisionFile string `yaml:"ProvisionFile"`
}

// LimitsConfig bounds resource usage across all jobs.
type LimitsConfig struct {
	// MaxJobs rejects devices beyond this count; 0 means unlimited.
	MaxJobs int `yaml:"MaxJobs"`
	// FrameTTLFactor scales the per-frame TTL against the frame interval.
	FrameTTLFactor float64 `yaml:"FrameTTLFactor"`
	// JSONPayloadCap bounds the size of structured JSON reading values.
	JSONPayloadCap int `yaml:"JSONPayloadCap"`
}

// New returns a config populated with safe defaults.
func New() InferenceConfig {
	return InferenceConfig{
		DefaultEndpoint: "edgex-ovms:9000",
		OpenAIEndpoint:  "http://edgex-vllm:8000",
		ModelRoot:       "/var/lib/nereid-inference/models",
		CoreMetadataURL: "http://edgex-core-metadata:59881",
		CoreCommandURL:  "http://edgex-core-command:59882",
		FFmpeg: FFmpegConfig{
			Bin:               "ffmpeg",
			ProbeBin:          "ffprobe",
			Global:            "-hide_banner -loglevel warning",
			ConnectTimeoutSec: 5,
			InputTemplates: map[string]string{
				"rtsp":     "-flags low_delay -max_delay 500000 -probesize 262144 -analyzeduration 1000000 -rtsp_transport tcp -timeout {timeout_us} -i {url}",
				"rtsp-udp": "-flags low_delay -max_delay 500000 -probesize 262144 -analyzeduration 1000000 -rtsp_transport udp -timeout {timeout_us} -i {url}",
			},
			OutputTemplate:         "-an -f rawvideo -pix_fmt rgb24 pipe:1",
			AuthChallengeNormalize: true,
			DropUID:                1000,
			DropGID:                1000,
			VAAPIDevice:            "/dev/dri/renderD128",
		},
		Onvif:   OnvifConfig{NormalizeVirtualChannel: true, MainChannel: "101"},
		Ess:     EssConfig{Enabled: true, LocalDir: "/var/horizon/ess-store/sync/local"},
		Preview: PreviewConfig{Enabled: true, Bind: "0.0.0.0:18080"},
		OVMS: OVMSConfig{
			Enabled:          true,
			RESTEndpoint:     "http://edgex-ovms:9001",
			ContainerRoot:    "/models",
			ReloadTimeoutSec: 5,
		},
		Triton: TritonConfig{
			Enabled:         false,
			ControlEndpoint: "http://edgex-triton:8000",
			GRPCEndpoint:    "edgex-triton:8001",
			ContainerRoot:   "/models",
			LoadTimeoutSec:  30,
			ReadyTimeoutSec: 60,
		},
		External: ExternalConfig{
			Enabled:       true,
			ProvisionFile: "res/external-servables.yaml",
		},
		Pipelines: PipelinesConfig{},
		Inbox:     InboxConfig{Enabled: true, Dir: "/var/lib/nereid-inference/inbox"},
		Limits:    LimitsConfig{MaxJobs: 64, FrameTTLFactor: 2.0, JSONPayloadCap: 1 << 20},
	}
}

// UpdateFromRaw replaces the configuration from a decoded section. It is
// called by the SDK both at initial load and on keeper updates.
func (c *InferenceConfig) UpdateFromRaw(rawConfig interface{}) bool {
	if rawConfig == nil {
		return false
	}
	b, err := json.Marshal(rawConfig)
	if err != nil {
		return false
	}
	next := New()
	if err := json.Unmarshal(b, &next); err != nil {
		return false
	}
	if !next.valid() {
		return false
	}
	*c = next
	return true
}

// Validate checks service-wide settings that cannot be repaired with
// defaults without silently changing operator intent.
func (c InferenceConfig) Validate() error {
	if c.ModelRoot == "" {
		return fmt.Errorf("ModelRoot must not be empty")
	}
	if c.Limits.MaxJobs < 0 {
		return fmt.Errorf("Limits.MaxJobs must be zero (unlimited) or positive")
	}
	if c.Limits.FrameTTLFactor <= 0 {
		return fmt.Errorf("Limits.FrameTTLFactor must be positive")
	}
	if c.Limits.JSONPayloadCap <= 0 {
		return fmt.Errorf("Limits.JSONPayloadCap must be positive")
	}
	if c.FFmpeg.ConnectTimeoutSec <= 0 {
		return fmt.Errorf("FFmpeg.ConnectTimeoutSec must be positive")
	}
	if c.FFmpeg.Bin == "" || c.FFmpeg.ProbeBin == "" || c.FFmpeg.OutputTemplate == "" {
		return fmt.Errorf("FFmpeg Bin, ProbeBin and OutputTemplate must not be empty")
	}
	if c.FFmpeg.DropUID < 0 || c.FFmpeg.DropGID < 0 {
		return fmt.Errorf("FFmpeg DropUID and DropGID must not be negative")
	}
	if c.FFmpeg.DropUID > 4294967295 || c.FFmpeg.DropGID > 4294967295 {
		return fmt.Errorf("FFmpeg DropUID and DropGID must fit in uint32")
	}
	if (c.FFmpeg.DropUID == 0) != (c.FFmpeg.DropGID == 0) {
		return fmt.Errorf("FFmpeg DropUID and DropGID must both be set or both be zero")
	}
	if c.FFmpeg.ConnectTimeoutSec > 60 {
		return fmt.Errorf("FFmpeg.ConnectTimeoutSec must be in 1..60")
	}
	if c.FFmpeg.VAAPIDevice != "" && !filepath.IsAbs(c.FFmpeg.VAAPIDevice) {
		return fmt.Errorf("FFmpeg.VAAPIDevice must be an absolute device path")
	}
	if c.Onvif.NormalizeVirtualChannel {
		channel := strings.TrimSpace(c.Onvif.MainChannel)
		if channel == "" || len(channel) > 16 || !channelTokenPattern.MatchString(channel) {
			return fmt.Errorf("Onvif.MainChannel must be a short channel token (default 101)")
		}
	}
	if c.OVMS.Enabled {
		if strings.TrimSpace(c.DefaultEndpoint) == "" {
			return fmt.Errorf("DefaultEndpoint is required when OVMS management is enabled")
		}
		if c.OVMS.RESTEndpoint == "" || c.OVMS.ContainerRoot == "" {
			return fmt.Errorf("OVMS RESTEndpoint and ContainerRoot are required when OVMS management is enabled")
		}
		if c.OVMS.ReloadTimeoutSec <= 0 || c.OVMS.ReloadTimeoutSec > 60 {
			return fmt.Errorf("OVMS.ReloadTimeoutSec must be in 1..60")
		}
	}
	if c.Triton.Enabled {
		if c.Triton.ControlEndpoint == "" || c.Triton.GRPCEndpoint == "" || c.Triton.ContainerRoot == "" {
			return fmt.Errorf("triton ControlEndpoint, GRPCEndpoint and ContainerRoot are required when Triton management is enabled")
		}
		if c.Triton.LoadTimeoutSec <= 0 || c.Triton.LoadTimeoutSec > 300 {
			return fmt.Errorf("Triton.LoadTimeoutSec must be in 1..300")
		}
		if c.Triton.ReadyTimeoutSec <= 0 || c.Triton.ReadyTimeoutSec > 600 {
			return fmt.Errorf("Triton.ReadyTimeoutSec must be in 1..600")
		}
	}
	if c.External.Enabled && c.External.ProvisionFile == "" {
		return fmt.Errorf("External.ProvisionFile must not be empty when the external registry is enabled")
	}
	return nil
}

func (c InferenceConfig) valid() bool { return c.Validate() == nil }
