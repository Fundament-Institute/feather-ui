# A render test that draws a circle.
using import .software-rendering
import .imageio
using import .circle-shader
using import glm

try
    # let circle = 
    #     texture.from-shader 256 256 vec4 circle-shader
    # imageio.save-into-file "output/circle.png" ('to-image circle)

    let circle-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                (circle (x / 256) (y / 256) (1 / 256) (vec4 .1 .1 .1 .1) (vec4 0.5 0.5 0.5 0.5) (vec4 0 1 0 1) (vec4 1 0 0 1))

    imageio.save-into-file "output/circle.png" ('to-image circle-render)

else
    print "Something failed while rendering circle-test"