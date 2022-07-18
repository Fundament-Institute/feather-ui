using import ..software-rendering
import ..imageio
using import .rectangle-shader
using import glm

try
    let rect-render = 
        texture.from-shader 256 256 vec4 
            fn (x y)
                let x = (f32 x)
                let y = (f32 y)
                (rect-shader x y 256.0 (vec4 256 256 8 0.5) (vec4 16 32 64 128) (vec4 1 1 1 1) (vec4 1 0 0 1))

    imageio.save-into-file "../output/rect.png" ('to-image rect-render)

else
    print "Something failed while rendering rect-test"