// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Package infer defines the vendor-neutral inference boundary and the two
// first-class backends: KServe gRPC v2 (vision tensors) and OpenAI-compatible
// HTTP (chat/embeddings/VLM). Backends are selected purely from the model
// descriptor runtime.backend; no vendor name ever appears in generic code.
package infer

import (
	"context"
	"fmt"
	"sync"
)

// TensorInput is one named input tensor.
type TensorInput struct {
	Name     string
	Shape    []int64
	Float32  []float32
	DataType string // FP32 only for the vision path
}

// TensorOutput is one decoded response tensor; only Float32 data is produced
// by the KServe vision path.
type TensorOutput struct {
	Name     string
	DataType string
	Shape    []int64
	Float32  []float32
}

// Request is a backend-neutral inference request. Messages handles the
// OpenAI chat/VLM path; Inputs handles the KServe tensor path.
type Request struct {
	ModelName    string
	ModelVersion string
	Inputs       []TensorInput
	Messages     []Message
	// SamplingJSON is a whitelist-validated JSON object passed through to
	// OpenAI APIs; the service never interprets its keys.
	SamplingJSON string
	Embeddings   []string
}

// Message is one OpenAI chat message (image_url carries a JPEG data URI for
// vision-language models).
type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// Response is the backend-neutral result.
type Response struct {
	Text       string
	Embeddings [][]float32
	Outputs    map[string]*TensorOutput
}

// Backend executes one inference request against one endpoint.
type Backend interface {
	Kind() string
	Ready(ctx context.Context, modelName, modelVersion string) error
	Infer(ctx context.Context, req Request) (*Response, error)
}

// Registry maintains one shared backend instance per endpoint, so gRPC
// channels and HTTP transports are reused across jobs.
type Registry struct {
	mu          sync.Mutex
	kserve      map[string]*KServeBackend
	openai      map[string]*OpenAIBackend
	openAIKeyFn func(secretName string) (map[string]string, error)
	apiSecret   string
}

// NewRegistry builds the backend registry. apiSecret names the EdgeX secret
// holding the OpenAI API key (may be empty on private networks).
func NewRegistry(apiSecret string, keyFn func(string) (map[string]string, error)) *Registry {
	return &Registry{
		kserve:      map[string]*KServeBackend{},
		openai:      map[string]*OpenAIBackend{},
		openAIKeyFn: keyFn,
		apiSecret:   apiSecret,
	}
}

// Get resolves the backend for a descriptor-selected protocol and endpoint.
func (r *Registry) Get(kind, endpoint string) (Backend, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch kind {
	case "kserve":
		if b, ok := r.kserve[endpoint]; ok {
			return b, nil
		}
		b, err := NewKServeBackend(endpoint)
		if err != nil {
			return nil, err
		}
		r.kserve[endpoint] = b
		return b, nil
	case "openai":
		if b, ok := r.openai[endpoint]; ok {
			return b, nil
		}
		var key string
		if r.apiSecret != "" && r.openAIKeyFn != nil {
			sec, err := r.openAIKeyFn(r.apiSecret)
			if err != nil {
				return nil, fmt.Errorf("get OpenAI API secret %q: %w", r.apiSecret, err)
			}
			key = first(sec["api_key"], sec["apiKey"], sec["Authorization"])
		}
		b, err := NewOpenAIBackend(endpoint, key)
		if err != nil {
			return nil, err
		}
		r.openai[endpoint] = b
		return b, nil
	default:
		return nil, fmt.Errorf("inference backend %q is not registered", kind)
	}
}

// Close releases shared connections.
func (r *Registry) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, b := range r.kserve {
		b.Close()
	}
	for _, b := range r.openai {
		b.Close()
	}
	r.kserve = map[string]*KServeBackend{}
	r.openai = map[string]*OpenAIBackend{}
}

func first(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
