// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

package decode

import (
	"github.com/edgexfoundry/device-inference/internal/preprocess"
)

// boundaryPolygon extracts the outer contour of a binary mask with a
// Moore-neighbor trace, normalizes the points into source-frame coordinates
// and stride-samples them to maxPoints. Masks are small (the prototype grid,
// typically 160x160), so the simple trace is cheap.
func boundaryPolygon(mask []bool, mh, mw, maxPoints int, meta preprocess.Meta) [][2]float32 {
	start := -1
	for i, v := range mask {
		if v {
			start = i
			break
		}
	}
	if start < 0 {
		return nil
	}
	sx, sy := start%mw, start/mw

	// Moore neighborhood, clockwise from west.
	dx := [8]int{-1, -1, 0, 1, 1, 1, 0, -1}
	dy := [8]int{0, -1, -1, -1, 0, 1, 1, 1}

	cx, cy := sx, sy
	entry := 6 // enter start from south
	var pts [][2]int
	firstVisit := true
	for {
		advanced := false
		for step := 0; step < 8; step++ {
			dir := (entry + step) % 8
			nx, ny := cx+dx[dir], cy+dy[dir]
			if nx < 0 || ny < 0 || nx >= mw || ny >= mh || !mask[ny*mw+nx] {
				continue
			}
			entry = (dir + 4 + 2) % 8
			pts = append(pts, [2]int{cx, cy})
			cx, cy = nx, ny
			advanced = true
			break
		}
		if !advanced {
			pts = append(pts, [2]int{cx, cy})
			break
		}
		if !firstVisit && cx == sx && cy == sy {
			break
		}
		firstVisit = false
		if len(pts) > mh*mw*4 {
			break
		}
	}
	if maxPoints > 0 && len(pts) > maxPoints {
		stride := len(pts) / maxPoints
		if stride < 1 {
			stride = 1
		}
		sampled := pts[:0]
		for i := 0; i < len(pts); i += stride {
			sampled = append(sampled, pts[i])
		}
		pts = sampled
	}
	out := make([][2]float32, 0, len(pts))
	for _, p := range pts {
		nx := float32(p[0]) / float32(mw)
		ny := float32(p[1]) / float32(mh)
		u, v, _, _ := meta.UnmapBox(nx, ny, nx, ny)
		out = append(out, [2]float32{u, v})
	}
	return out
}
