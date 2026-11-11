// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package pipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
)

// DecodeDefinitionsJSON decodes the exact PipelinesJSON representation. A
// present empty array is valid; null, non-arrays, trailing values, unknown or
// incorrectly-cased fields are not silently normalized into another table.
func DecodeDefinitionsJSON(data []byte) ([]Definition, error) {
	var records []Definition
	if err := decodeStrictJSON(data, &records); err != nil {
		return nil, err
	}
	if records == nil {
		return nil, fmt.Errorf("pipelines table must be a JSON array, not null")
	}
	return records, nil
}

func decodeStoredRegistryJSON(data []byte) (storedRegistry, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		records, err := DecodeDefinitionsJSON(trimmed)
		if err != nil {
			return storedRegistry{}, err
		}
		// Original development-format arrays had no persisted history. The next
		// successful replacement writes the complete history envelope.
		return storedRegistry{Pipelines: records, History: map[string]string{}}, nil
	}
	var state storedRegistry
	if err := decodeStrictJSON(trimmed, &state); err != nil {
		return storedRegistry{}, err
	}
	if state.Pipelines == nil {
		return storedRegistry{}, fmt.Errorf("persisted pipelines must contain a non-null pipelines array")
	}
	if state.History == nil {
		return storedRegistry{}, fmt.Errorf("persisted pipelines must contain a non-null history object")
	}
	return state, nil
}

// decodeStrictJSON combines encoding/json's type checking with an exact-key
// walk. DisallowUnknownFields alone accepts case-insensitive aliases such as
// PipelineId for pipelineId, which would make the wire schema ambiguous.
func decodeStrictJSON(data []byte, out any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(out); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values are not allowed")
		}
		return fmt.Errorf("invalid trailing JSON: %w", err)
	}

	rawDecoder := json.NewDecoder(bytes.NewReader(data))
	rawDecoder.UseNumber()
	var raw any
	if err := rawDecoder.Decode(&raw); err != nil {
		return err
	}
	return validateExactJSONFields(raw, reflect.TypeOf(out).Elem(), "")
}

func validateExactJSONFields(value any, typ reflect.Type, path string) error {
	if value == nil {
		return nil
	}
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch typ.Kind() {
	case reflect.Struct:
		object, ok := value.(map[string]any)
		if !ok {
			return nil // The typed decoder above reports the type error.
		}
		fields := make(map[string]reflect.Type, typ.NumField())
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			if field.PkgPath != "" {
				continue
			}
			name := strings.Split(field.Tag.Get("json"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = field.Name
			}
			fields[name] = field.Type
		}
		for name, member := range object {
			fieldType, ok := fields[name]
			if !ok {
				if path == "" {
					return fmt.Errorf("json: unknown field %q", name)
				}
				return fmt.Errorf("json: unknown field %q in %s", name, path)
			}
			memberPath := name
			if path != "" {
				memberPath = path + "." + name
			}
			if err := validateExactJSONFields(member, fieldType, memberPath); err != nil {
				return err
			}
		}
	case reflect.Slice, reflect.Array:
		// []byte has a string JSON representation, not an array of numbers.
		if typ.Elem().Kind() == reflect.Uint8 {
			return nil
		}
		array, ok := value.([]any)
		if !ok {
			return nil
		}
		for i, member := range array {
			if err := validateExactJSONFields(member, typ.Elem(), fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	}
	return nil
}
