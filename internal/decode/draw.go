// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"image"
	"image/color"
	"strconv"
	"unicode"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

var highContrastPalette = []color.RGBA{
	{R: 0, G: 220, B: 70, A: 255},   // green
	{R: 255, G: 210, B: 0, A: 255},  // yellow
	{R: 0, G: 210, B: 255, A: 255},  // cyan
	{R: 255, G: 120, B: 0, A: 255},  // orange
	{R: 255, G: 0, B: 180, A: 255},  // magenta
	{R: 150, G: 255, B: 0, A: 255},  // lime
	{R: 255, G: 70, B: 70, A: 255},  // red
	{R: 90, G: 140, B: 255, A: 255}, // blue
}

// palette deterministically assigns a bright, surveillance-preview-friendly
// RGB triple to a class id. Muted random colors are difficult to distinguish
// on noisy camera frames, particularly after JPEG/MJPEG compression.
func palette(id int) color.RGBA {
	if id < 0 {
		id = -id
	}
	return highContrastPalette[id%len(highContrastPalette)]
}

func drawBox(canvas *image.RGBA, box [4]float32, class int, label string, score float32) {
	b := canvas.Bounds()
	x0 := int(box[0] * float32(b.Dx()))
	y0 := int(box[1] * float32(b.Dy()))
	x1 := int(box[2] * float32(b.Dx()))
	y1 := int(box[3] * float32(b.Dy()))
	c := palette(class)
	rect(canvas, x0, y0, x1, y1, c)
	caption(canvas, x0, y0, label+" "+strconv.FormatFloat(float64(score), 'f', 2, 32))
}

func rect(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	for x := x0; x <= x1; x++ {
		img.Set(x, y0, c)
		img.Set(x, y1, c)
	}
	for y := y0; y <= y1; y++ {
		img.Set(x0, y, c)
		img.Set(x1, y, c)
	}
}

func fillCell(img *image.RGBA, x0, y0, x1, y1 int, plot func(x, y int)) {
	b := img.Bounds()
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x1 >= b.Max.X {
		x1 = b.Max.X - 1
	}
	if y1 >= b.Max.Y {
		y1 = b.Max.Y - 1
	}
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			plot(x, y)
		}
	}
}

func caption(img *image.RGBA, x, y int, text string) {
	if y < 12 {
		y += 12
	}
	if !isASCII(text) {
		// Built-in bitmap font is ASCII only; CJK labels render as ids by
		// convention (mount a TTF for localized previews in later releases).
		return
	}
	width := font.MeasureString(basicfont.Face7x13, text).Ceil() + 4
	fillRect(img, x, y-11, x+width, y+2, image.White)
	d := &font.Drawer{
		Dst:  img,
		Src:  image.Black,
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x+2, y-1),
	}
	d.DrawString(text)
}

func isASCII(s string) bool {
	for _, r := range s {
		if r > unicode.MaxASCII {
			return false
		}
	}
	return true
}

func fillRect(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			img.Set(x, y, c)
		}
	}
}

// drawLine connects two normalized points (Bresenham).
func drawLine(img *image.RGBA, a, b Keypoint) {
	bounds := img.Bounds()
	x0 := int(a.X * float32(bounds.Dx()))
	y0 := int(a.Y * float32(bounds.Dy()))
	x1 := int(b.X * float32(bounds.Dx()))
	y1 := int(b.Y * float32(bounds.Dy()))
	dx := absI(x1 - x0)
	dy := -absI(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	c := color.RGBA{R: 60, G: 220, B: 120, A: 255}
	for {
		img.Set(x0, y0, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func absI(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
