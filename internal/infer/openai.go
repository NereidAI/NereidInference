// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package infer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// OpenAIBackend talks to any OpenAI-compatible endpoint (vLLM, OVMS-LLM,
// MindIE). Sampling parameters are whitelist-forwarded by the caller; the
// DS itself never interprets them.
type OpenAIBackend struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// maxOpenAIResponse bounds a chat or embeddings response from a model
// service; text outputs are deliberately bounded separately upstream.
const maxOpenAIResponse = 32 * 1024 * 1024

// NewOpenAIBackend constructs the backend.
func NewOpenAIBackend(baseURL, apiKey string) (*OpenAIBackend, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid OpenAI endpoint %q: %w", baseURL, err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf("invalid OpenAI endpoint %q", baseURL)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	return &OpenAIBackend{
		baseURL: baseURL,
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 120 * time.Second, Transport: transport},
	}, nil
}

// Close releases idle HTTP transport connections.
func (b *OpenAIBackend) Close() {
	if t, ok := b.http.Transport.(*http.Transport); ok {
		t.CloseIdleConnections()
	}
}

// Kind returns the backend kind.
func (b *OpenAIBackend) Kind() string { return "openai" }

// Ready probes GET /v1/models and verifies that the requested model is
// advertised by the OpenAI-compatible service.
func (b *OpenAIBackend) Ready(ctx context.Context, modelName, _ string) error {
	rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, b.baseURL+"/v1/models", nil)
	if err != nil {
		return err
	}
	b.authorize(req)
	resp, err := b.http.Do(req)
	if err != nil {
		return err
	}
	defer drainClose(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("openai backend not ready: HTTP %d", resp.StatusCode)
	}
	var listed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxOpenAIResponse)).Decode(&listed); err != nil {
		return err
	}
	for _, model := range listed.Data {
		if model.ID == modelName {
			return nil
		}
	}
	return fmt.Errorf("openai model %q is not advertised", modelName)
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Infer dispatches chat/completions or embeddings based on the request
// payload.
func (b *OpenAIBackend) Infer(ctx context.Context, req Request) (*Response, error) {
	switch {
	case len(req.Embeddings) > 0:
		return b.embeddings(ctx, req)
	case len(req.Messages) > 0:
		return b.chat(ctx, req)
	default:
		return nil, errors.New("openai request has neither messages nor embeddings")
	}
}

func (b *OpenAIBackend) chat(ctx context.Context, req Request) (*Response, error) {
	body := map[string]any{
		"model":    req.ModelName,
		"messages": req.Messages,
	}
	if err := mergeSampling(body, req.SamplingJSON); err != nil {
		return nil, err
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, b.baseURL+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	b.authorize(httpReq)
	resp, err := b.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer drainClose(resp.Body)
	payload, err := readLimited(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("chat completions HTTP %d: %s", resp.StatusCode, truncate(payload))
	}
	var cr chatResponse
	if err := json.Unmarshal(payload, &cr); err != nil {
		return nil, err
	}
	if len(cr.Choices) == 0 {
		return nil, errors.New("chat completions returned no choices")
	}
	return &Response{Text: cr.Choices[0].Message.Content}, nil
}

func (b *OpenAIBackend) embeddings(ctx context.Context, req Request) (*Response, error) {
	body := map[string]any{"model": req.ModelName, "input": req.Embeddings}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, b.baseURL+"/v1/embeddings", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	b.authorize(httpReq)
	resp, err := b.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer drainClose(resp.Body)
	payload, err := readLimited(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("embeddings HTTP %d: %s", resp.StatusCode, truncate(payload))
	}
	var cr chatResponse
	if err := json.Unmarshal(payload, &cr); err != nil {
		return nil, err
	}
	if len(cr.Data) != len(req.Embeddings) {
		return nil, fmt.Errorf("embeddings returned %d vectors for %d inputs", len(cr.Data), len(req.Embeddings))
	}
	out := &Response{Embeddings: make([][]float32, len(cr.Data))}
	seen := make([]bool, len(cr.Data))
	for _, d := range cr.Data {
		if d.Index < 0 || d.Index >= len(out.Embeddings) {
			return nil, fmt.Errorf("embedding index %d out of range", d.Index)
		}
		if seen[d.Index] {
			return nil, fmt.Errorf("duplicate embedding index %d", d.Index)
		}
		if len(d.Embedding) == 0 {
			return nil, fmt.Errorf("embedding index %d is empty", d.Index)
		}
		seen[d.Index] = true
		out.Embeddings[d.Index] = d.Embedding
	}
	return out, nil
}

func (b *OpenAIBackend) authorize(req *http.Request) {
	if b.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
	}
}

func mergeSampling(body map[string]any, rawJSON string) error {
	if rawJSON == "" {
		return nil
	}
	var extra map[string]any
	if err := json.Unmarshal([]byte(rawJSON), &extra); err != nil {
		return fmt.Errorf("invalid sampling parameters: %w", err)
	}
	for key, value := range extra {
		switch key {
		case "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
			"frequency_penalty", "presence_penalty", "stop", "seed", "n",
			"logit_bias", "response_format":
			body[key] = value
		default:
			return fmt.Errorf("sampling parameter %q is not permitted", key)
		}
	}
	return nil
}

func truncate(b []byte) string {
	const max = 256
	if len(b) > max {
		return string(b[:max]) + "..."
	}
	return string(b)
}

func drainClose(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 1024))
	_ = body.Close()
}

func readLimited(r io.Reader) ([]byte, error) {
	payload, err := io.ReadAll(io.LimitReader(r, maxOpenAIResponse+1))
	if err != nil {
		return nil, err
	}
	if int64(len(payload)) > maxOpenAIResponse {
		return nil, fmt.Errorf("OpenAI response exceeds %d bytes", maxOpenAIResponse)
	}
	return payload, nil
}
