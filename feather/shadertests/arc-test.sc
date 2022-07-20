# A render test that draws an arc.
using import ..software-rendering
import ..imageio
using import .arc-shader
using import .shader-utils
using import glm

try
    let arc-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                let ang-a = 1
                let ang-b = 1
                let radius = 64
                (arc x y 256.0 (vec4 1 1 4 1.0) (vec4 (ang-a + (ang-b / 2.0) - (pi / 2.0)) ang-b radius 0) (vec4 1 1 1 1) (vec4 1 0 0 1))

    imageio.save-into-file "../output/arc.png" ('to-image arc-render)

else
    print "Something failed while rendering arc-test"