using import glm
using import glsl

fn line (x y color center)
    let uv = (vec2 x y)
    local col = color
    let d = (length (center - uv))
    let width = 1.0
    col.a = (col.a * (clamp 0.0 1.0 (width - d)))
    (vec4 (col.r * col.a) (col.g * col.a) (col.b * col.a) col.a)

do
    let line
    locals;