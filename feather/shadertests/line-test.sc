# A render test that draws a line.
using import ..software-rendering
import ..imageio
using import .line-shader
using import glm

try
    let line-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                # (old-line x y (vec4 1 1 1 1) (vec2 128.0 128.0))
                (line x y (vec4 1) (vec4 0 0 1.0 1.0))

    imageio.save-into-file "../output/line.png" ('to-image line-render)

else
    print "Something failed while rendering line-test"