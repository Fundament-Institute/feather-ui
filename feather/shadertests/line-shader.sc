using import glm
using import glsl
using import .shader-utils

fn old-line (x y color center)
    # Not currently working. I'm not 100% sure what the deal is.
    let uv = (vec2 x y)
    local col = color
    let d = (length (center - uv))
    let width = 1.0
    col.a = (col.a * (clamp 0.0 1.0 (width - d)))
    (vec4 (col.r * col.a) (col.g * col.a) (col.b * col.a) col.a)

fn line (x y color PosWidthBlur)
    # Dead simple line. Only draws diagonally for now.
    let uv = (vec2 (x + 0.5) (y + 0.5)) # add half-pixel
    let w = (1.0 + PosWidthBlur.w) # Todo: blur param
    let o = (min uv.x uv.y)
    let dist = ((sqrt ((pow (uv.x - o) 2) + (pow (uv.y - x) 2))) - PosWidthBlur.z)
    let alpha = (pow (linearstep w 0.0 dist) 2.2)
    (vec4 color.rgb alpha)

do
    let line
    let old-line
    locals;