// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"image/color"
	"testing"
)

func TestDetectionPaletteUsesHighContrastColors(t *testing.T) {
	green := color.RGBA{R: 0, G: 220, B: 70, A: 255}
	yellow := color.RGBA{R: 255, G: 210, B: 0, A: 255}
	if got := palette(0); got != green {
		t.Fatalf("class 0 color = %+v, want green %+v", got, green)
	}
	if got := palette(1); got != yellow {
		t.Fatalf("class 1 color = %+v, want yellow %+v", got, yellow)
	}
	if got := palette(len(highContrastPalette)); got != green {
		t.Fatalf("palette does not cycle deterministically: %+v", got)
	}
}
