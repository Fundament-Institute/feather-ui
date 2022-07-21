# A render test that draws a triangle.
using import ..software-rendering
import ..imageio
using import .triangle-shader
using import glm

try
    let triangle-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                (triangle x y 256.0 (vec4 256 256 8.0 0.5) (vec4 0 0 0 0.5) (vec4 1 1 1 1) (vec4 0 0.5 1 1))

    imageio.save-into-file "../output/triangle.png" ('to-image triangle-render)

else
    print "Something failed while rendering triangle-test"