using import glm
using import glsl
using import .shader-utils

fn... triangle (x : f32, y : f32, size : f32, DimBorderBlur : vec4, corners : vec4, fill : vec4, outline : vec4)
    let d = DimBorderBlur.xy
    let p = (((vec2 (x + 0.5) (y + 0.5)) * 2.0) - (vec2 (size - (d / 2))))
    let p2 = (vec2 (corners.w * d.x) 0.0)
    let r1 = (point-line-side p2 (vec2 0.0 d.y) p)
    let r2 = (- (point-line-side p2 d p))
    let r = (max (max r1 r2) (p.y - d.y)) 
    # Don't ask me why *this* works but the original border calculation doesn't. This was trial and error.
    let b1 = (point-line-side p2 (vec2 0.0 d.y) p)
    let b2 = (point-line-side d p2 p)
    let b = (max b1 b2 (p.y - d.y) (p.x - d.x - DimBorderBlur.z))

    let w = (1.0 + DimBorderBlur.w)
    let alpha = (linearstep w (- w) (b - (2.0 * DimBorderBlur.z)))
    let s = (linearstep w (- w) r)
    let tri = ((vec4 fill.rgb 1.0) * fill.a * s)
    let outl = ((vec4 outline.rgb 1.0) * outline.a * (clamp (alpha - s) 0.0 1.0))
    let output = (tri + outl)
    output

do
    let triangle
    locals;