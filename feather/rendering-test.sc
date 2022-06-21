import .imageio
using import Array
using import struct
using import Capture
using import glm
using import .software-rendering

# declare the vertex type for drawing colored triangles
struct vertex
    pos : vec3
    col : vec3

    inline __lerp (a b x)
        this-type
            pos = (lerp a.pos b.pos x)
            col = (lerp a.col b.col x)

try

    # synthesize a gradient with a shader
    let gradient =
        texture.from-shader 256 256 vec4
            fn (x y)
                vec4 (x / 256) (y / 256) 0.5 1.0

    # alpha-blend a different texture on top of the gradient
    # let tg =
        texture.from-shader 256 256 vec4
            capture (x y) {(view gradient) (view tex)}
                let gp tp = ('@ gradient x y) ('@ tex x y)
                let a = (tp.a + gp.a * (1 - tp.a))
                vec4
                    (tp.rgb * tp.a + gp.rgb * gp.a * (1 - tp.a)) / a
                    a

    # construct a texture by using draw calls
    let res =
        texture.from-drawing 256 256 vec3
            capture (c) {}
                'clear c (vec3 0.2 0.3 0)
                # rasterize a triangle
                'triangle c
                    vertex
                        pos = (vec3 100 100 100)
                        col = (vec3 1 0 0)
                    vertex
                        pos = (vec3 200 100 100)
                        col = (vec3 0 1 0)
                    vertex
                        pos = (vec3 150 200 100)
                        col = (vec3 0 0 1)
                    # pixel shader to draw the triangle with
                    inline (point)
                        point.col
                'triangle c
                    vertex
                        pos = (vec3 50 50 50)
                        col = (vec3 1 0 0)
                    vertex
                        pos = (vec3 150 50 110)
                        col = (vec3 0 1 0)
                    vertex
                        pos = (vec3 100 150 100)
                        col = (vec3 0 0 1)
                    inline (point)
                        point.col

    res :=
        'map res vec4
            inline (px)
                vec4 px 1

    imageio.save-into-file "output/gradient.png" ('to-image gradient)

    let res = ('to-image res)
    print res
    print res.width
    print res.height
    print res.pixel_format
    print res.bytes_per_line
    imageio.save-into-file "output/res.png" res
    print "saved successfully"

else
    print "something failed"
