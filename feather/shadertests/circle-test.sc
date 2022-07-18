# A render test that draws a circle.
using import ..software-rendering
import ..imageio
using import .circle-shader
using import glm

try
    # let circle = 
    #     texture.from-shader 256 256 vec4 circle-shader
    # imageio.save-into-file "output/circle.png" ('to-image circle)

    let circle-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                (circle x y 256.0 (vec4 128 128 4 1.5) (vec4 0 0 0 0) (vec4 1 1 1 1) (vec4 1 0 0 1))

    imageio.save-into-file "../output/circle.png" ('to-image circle-render)

else
    print "Something failed while rendering circle-test"