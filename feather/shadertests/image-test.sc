# A render test that draws an arbitrary image.
using import ..software-rendering
import ..imageio
using import .image-shader
using import enum
using import Capture
using import glm

enum image-shader-error
    Error
    imageio.raw.enum.SailStatus

try
    let img = (imageio.load-from-file "./sample.png")
    let image-render = 
        texture.from-shader 256 256 vec4 
            capture f (x y) {img}
                try
                    (image-shader x y 256 img)
                except (e)
                    raise (tostring e)

    imageio.save-into-file "../output/image.png" ('to-image image-render)

else
    print "Something failed while rendering image-test"