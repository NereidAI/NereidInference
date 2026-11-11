// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package infer

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"time"

	pb "github.com/edgexfoundry/device-inference/internal/infer/kserve/grpcv2"
	"github.com/edgexfoundry/device-inference/internal/model"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// KServeBackend speaks the official KServe Predict Protocol v2 over a
// persistent HTTP/2 gRPC channel. REST is intentionally not used: P0
// measured roughly 50 ms/frame of JSON/HTTP overhead, unusable for
// continuous video.
const maxKServeRecvBytes = 256 * 1024 * 1024

type KServeBackend struct {
	endpoint string
	conn     *grpc.ClientConn
	client   pb.GRPCInferenceServiceClient
}

// NewKServeBackend establishes a lazy dial; the Ready probe verifies
// reachability before a job is admitted.
func NewKServeBackend(endpoint string) (*KServeBackend, error) {
	if endpoint == "" {
		return nil, errors.New("kserve endpoint is empty")
	}
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		// Vision heads (segmentation, image tensors) routinely exceed the
		// gRPC default 4 MiB receive limit. Bound memory at 256 MiB instead.
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxKServeRecvBytes),
			grpc.MaxCallSendMsgSize(maxKServeRecvBytes),
		),
	)
	if err != nil {
		return nil, err
	}
	return &KServeBackend{
		endpoint: endpoint,
		conn:     conn,
		client:   pb.NewGRPCInferenceServiceClient(conn),
	}, nil
}

// Kind returns the backend kind.
func (b *KServeBackend) Kind() string { return "kserve" }

// Ready calls ModelReady for the given model.
func (b *KServeBackend) Ready(ctx context.Context, modelName, version string) error {
	rctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	resp, err := b.client.ModelReady(rctx, &pb.ModelReadyRequest{Name: modelName, Version: version})
	if err != nil {
		return err
	}
	if !resp.GetReady() {
		return status.Error(codes.NotFound, "model "+modelName+" is not ready")
	}
	return nil
}

// Metadata fetches model I/O signatures for descriptor cross-checks.
func (b *KServeBackend) Metadata(ctx context.Context, modelName, version string) ([]model.IOSpec, []model.IOSpec, error) {
	rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	resp, err := b.client.ModelMetadata(rctx, &pb.ModelMetadataRequest{Name: modelName, Version: version})
	if err != nil {
		return nil, nil, err
	}
	in := make([]model.IOSpec, 0, len(resp.GetInputs()))
	for _, t := range resp.GetInputs() {
		in = append(in, tensorToSpec(t))
	}
	out := make([]model.IOSpec, 0, len(resp.GetOutputs()))
	for _, t := range resp.GetOutputs() {
		out = append(out, tensorToSpec(t))
	}
	return in, out, nil
}

// Infer performs one unary ModelInfer. FP32 tensors ride the protobuf packed
// representation, so no manual wire encoding is needed on gRPC; servers may
// additionally answer through raw_output_contents (binary extension).
func (b *KServeBackend) Infer(ctx context.Context, req Request) (*Response, error) {
	inputs := make([]*pb.ModelInferRequest_InferInputTensor, 0, len(req.Inputs))
	for _, t := range req.Inputs {
		inputs = append(inputs, &pb.ModelInferRequest_InferInputTensor{
			Name:     t.Name,
			Datatype: "FP32",
			Shape:    t.Shape,
			Contents: &pb.InferTensorContents{Fp32Contents: t.Float32},
		})
	}
	resp, err := b.client.ModelInfer(ctx, &pb.ModelInferRequest{
		ModelName:    req.ModelName,
		ModelVersion: req.ModelVersion,
		Inputs:       inputs,
	})
	if err != nil {
		return nil, err
	}
	return parseKServeResponse(resp)
}

func parseKServeResponse(resp *pb.ModelInferResponse) (*Response, error) {
	outputs := map[string]*TensorOutput{}
	useRaw := len(resp.GetRawOutputContents()) == len(resp.GetOutputs())
	for i, t := range resp.GetOutputs() {
		to := &TensorOutput{Name: t.Name, DataType: t.Datatype, Shape: t.Shape}
		switch {
		case useRaw && len(resp.RawOutputContents[i]) > 0:
			values, err := decodeRawTensor(t.Datatype, resp.RawOutputContents[i])
			if err != nil {
				return nil, fmt.Errorf("output %q: %w", t.Name, err)
			}
			to.Float32 = values
		case t.Contents != nil:
			values, err := decodeTypedTensor(t.Datatype, t.Contents)
			if err != nil {
				return nil, fmt.Errorf("output %q: %w", t.Name, err)
			}
			to.Float32 = values
		default:
			return nil, fmt.Errorf("output %q has neither raw nor typed contents", t.Name)
		}
		if expected, ok := shapeElements(t.Shape); ok && len(to.Float32) != expected {
			return nil, fmt.Errorf("output %q has %d values, shape requires %d", t.Name, len(to.Float32), expected)
		}
		outputs[t.Name] = to
	}
	return &Response{Outputs: outputs}, nil
}

func shapeElements(shape []int64) (int, bool) {
	var n int64 = 1
	for _, dim := range shape {
		if dim <= 0 {
			// Dynamic/unknown dimension: exact length cannot be checked.
			return 0, false
		}
		if n > math.MaxInt64/dim {
			return 0, false
		}
		n *= dim
	}
	if n > math.MaxInt {
		return 0, false
	}
	return int(n), true
}

func decodeTypedTensor(datatype string, c *pb.InferTensorContents) ([]float32, error) {
	switch datatype {
	case "FP32":
		return append([]float32(nil), c.GetFp32Contents()...), nil
	case "FP64":
		return float64To32(c.GetFp64Contents()), nil
	case "INT8", "INT16", "INT32":
		out := make([]float32, len(c.GetIntContents()))
		for i, v := range c.GetIntContents() {
			out[i] = float32(v)
		}
		return out, nil
	case "INT64":
		out := make([]float32, len(c.GetInt64Contents()))
		for i, v := range c.GetInt64Contents() {
			out[i] = float32(v)
		}
		return out, nil
	case "UINT8", "UINT16", "UINT32":
		out := make([]float32, len(c.GetUintContents()))
		for i, v := range c.GetUintContents() {
			out[i] = float32(v)
		}
		return out, nil
	case "UINT64":
		out := make([]float32, len(c.GetUint64Contents()))
		for i, v := range c.GetUint64Contents() {
			out[i] = float32(v)
		}
		return out, nil
	case "BOOL":
		out := make([]float32, len(c.GetBoolContents()))
		for i, v := range c.GetBoolContents() {
			if v {
				out[i] = 1
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported response datatype %q", datatype)
	}
}

func decodeRawTensor(datatype string, raw []byte) ([]float32, error) {
	var width int
	switch datatype {
	case "BOOL", "UINT8", "INT8":
		width = 1
	case "UINT16", "INT16":
		width = 2
	case "UINT32", "INT32", "FP32":
		width = 4
	case "UINT64", "INT64", "FP64":
		width = 8
	default:
		return nil, fmt.Errorf("unsupported raw response datatype %q", datatype)
	}
	if len(raw)%width != 0 {
		return nil, fmt.Errorf("%s payload length %d is not divisible by %d", datatype, len(raw), width)
	}
	n := len(raw) / width
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		off := i * width
		switch datatype {
		case "BOOL":
			if raw[off] != 0 {
				out[i] = 1
			}
		case "UINT8":
			out[i] = float32(raw[off])
		case "INT8":
			v := int(raw[off])
			if v > 127 {
				v -= 256
			}
			out[i] = float32(v)
		case "UINT16":
			out[i] = float32(binary.LittleEndian.Uint16(raw[off:]))
		case "INT16":
			v := int(binary.LittleEndian.Uint16(raw[off:]))
			if v > 32767 {
				v -= 65536
			}
			out[i] = float32(v)
		case "UINT32":
			out[i] = float32(binary.LittleEndian.Uint32(raw[off:]))
		case "INT32":
			v := int64(binary.LittleEndian.Uint32(raw[off:]))
			if v > math.MaxInt32 {
				v -= 1 << 32
			}
			out[i] = float32(v)
		case "FP32":
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[off:]))
		case "UINT64":
			out[i] = float32(binary.LittleEndian.Uint64(raw[off:]))
		case "INT64":
			v := binary.LittleEndian.Uint64(raw[off:])
			if v <= math.MaxInt64 {
				out[i] = float32(v)
			} else {
				out[i] = float32(float64(v-(1<<63)) - float64(1<<63))
			}
		case "FP64":
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[off:])))
		}
	}
	return out, nil
}

func float64To32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func tensorToSpec(t *pb.ModelMetadataResponse_TensorMetadata) model.IOSpec {
	return model.IOSpec{Name: t.Name, Shape: t.Shape, DType: t.Datatype}
}

// Close terminates the shared channel.
func (b *KServeBackend) Close() {
	if b.conn != nil {
		_ = b.conn.Close()
	}
}
