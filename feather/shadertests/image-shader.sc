using import glm
using import glsl
using import .shader-utils
using import ..software-rendering
import ..imageio

fn... image-shader (x : i32, y : i32, size : i32, img : imageio.image)
    let tex = (texture.from-image img)
    let dims = (vec2 tex.size.x tex.size.y)
    let xy = (vec2 (x / size) (y / size))
    let uv = (xy * dims)
    let coord = ((copy uv.x) + ((copy uv.y) * (copy dims.y)))
    (copy ('@ tex uv.x uv.y))

do
    let image-shader
    locals;
