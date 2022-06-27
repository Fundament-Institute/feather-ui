# A render test that draws a line.
using import .software-rendering
import .imageio
using import .line-shader
using import glm

try
    let line-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                (line (x / 255.) (y / 255.) (vec4 1 1 1 1) (vec2 .5 .5))

    imageio.save-into-file "output/line.png" ('to-image line-render)

else
    print "Something failed while rendering line-test"