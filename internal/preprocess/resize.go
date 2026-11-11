// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package preprocess

import (
	"image"
)

// resizeBilinear scales src to dstW x dstH using standard bilinear
// interpolation on the RGBA representation. It operates on modest input
// sizes (camera frames at single-digit job FPS), which is comfortably
// covered by pure Go.
func resizeBilinear(src image.Image, dstW, dstH int) *image.RGBA {
	b := src.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	if srcW == 0 || srcH == 0 {
		return dst
	}
	xRatio := float32(srcW) / float32(dstW)
	yRatio := float32(srcH) / float32(dstH)
	for y := 0; y < dstH; y++ {
		sy := (float32(y)+0.5)*yRatio - 0.5
		y0 := intF(sy)
		dy := sy - float32(y0)
		if y0 < 0 {
			y0, dy = 0, 0
		}
		y1 := y0 + 1
		if y1 > srcH-1 {
			y1 = srcH - 1
		}
		for x := 0; x < dstW; x++ {
			sx := (float32(x)+0.5)*xRatio - 0.5
			x0 := intF(sx)
			dx := sx - float32(x0)
			if x0 < 0 {
				x0, dx = 0, 0
			}
			x1 := x0 + 1
			if x1 > srcW-1 {
				x1 = srcW - 1
			}
			c00 := atRGB(src, b.Min.X+x0, b.Min.Y+y0)
			c10 := atRGB(src, b.Min.X+x1, b.Min.Y+y0)
			c01 := atRGB(src, b.Min.X+x0, b.Min.Y+y1)
			c11 := atRGB(src, b.Min.X+x1, b.Min.Y+y1)
			off := dst.PixOffset(x, y)
			dst.Pix[off+0] = lerpU8(lerpU8(c00[0], c10[0], dx), lerpU8(c01[0], c11[0], dx), dy)
			dst.Pix[off+1] = lerpU8(lerpU8(c00[1], c10[1], dx), lerpU8(c01[1], c11[1], dx), dy)
			dst.Pix[off+2] = lerpU8(lerpU8(c00[2], c10[2], dx), lerpU8(c01[2], c11[2], dx), dy)
			dst.Pix[off+3] = 255
		}
	}
	return dst
}

func atRGB(img image.Image, x, y int) [3]uint8 {
	if rgba, ok := img.(*image.RGBA); ok {
		off := rgba.PixOffset(x, y)
		return [3]uint8{rgba.Pix[off], rgba.Pix[off+1], rgba.Pix[off+2]}
	}
	r, g, b, _ := img.At(x, y).RGBA()
	return [3]uint8{quantize16(r), quantize16(g), quantize16(b)}
}

// quantize16 converts a color alpha-premultiplied 16-bit channel to 8 bits.
// Division keeps the result in 0..255 without a truncating conversion.
func quantize16(v uint32) uint8 {
	return uint8(v / 257) //nolint:gosec // v is a 16-bit color channel, so v/257 cannot exceed 255.
}

func intF(v float32) int { return int(v) }

func lerpU8(a, b uint8, t float32) uint8 {
	v := float32(a)*(1-t) + float32(b)*t
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5) //nolint:gosec // v is explicitly constrained to 0..255.
}

func u8Raw(v float32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5) //nolint:gosec // v is explicitly constrained to 0..255.
}

// fillRGB paints an RGBA canvas with raw 0..255 channel values. Descriptor
// normalization is applied uniformly to both image and pad pixels later.
func fillRGB(dst *image.RGBA, r, g, b float32) {
	rv, gv, bv := u8Raw(r), u8Raw(g), u8Raw(b)
	for i := 0; i < len(dst.Pix); i += 4 {
		dst.Pix[i] = rv
		dst.Pix[i+1] = gv
		dst.Pix[i+2] = bv
		dst.Pix[i+3] = 255
	}
}

// copyRect pastes src onto dst at (offX, offY) without clipping outside.
func copyRect(dst *image.RGBA, src *image.RGBA, offX, offY int) {
	b := src.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			if offX+x >= dst.Bounds().Dx() || offY+y >= dst.Bounds().Dy() {
				continue
			}
			sOff := src.PixOffset(b.Min.X+x, b.Min.Y+y)
			dOff := dst.PixOffset(offX+x, offY+y)
			copy(dst.Pix[dOff:dOff+4], src.Pix[sOff:sOff+4])
		}
	}
}

// copyCenteredCrop copies the central dst-sized region from src when the
// offsets are negative (fill resize followed by center crop).
func copyCenteredCrop(dst *image.RGBA, src *image.RGBA, offX, offY int) {
	if offX >= 0 && offY >= 0 {
		copyRect(dst, src, offX, offY)
		return
	}
	db := dst.Bounds()
	sb := src.Bounds()
	for y := db.Min.Y; y < db.Max.Y; y++ {
		sy := sb.Min.Y + y - offY
		if sy < sb.Min.Y || sy >= sb.Max.Y {
			continue
		}
		for x := db.Min.X; x < db.Max.X; x++ {
			sx := sb.Min.X + x - offX
			if sx < sb.Min.X || sx >= sb.Max.X {
				continue
			}
			sOff := src.PixOffset(sx, sy)
			dOff := dst.PixOffset(x, y)
			copy(dst.Pix[dOff:dOff+4], src.Pix[sOff:sOff+4])
		}
	}
}
