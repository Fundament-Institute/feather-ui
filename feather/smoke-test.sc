

using import .backend
using import glm

let levels =
    arrayof rawstring
        "Fatal: "
        "Error: "
        "Warning: "
        "Notice: "
        "Debug: "

        #FIXME no way to write a log function here.

fn mat4-proj (l r b t n f)
    """"assembles a custom projection matrix designed for 2d drawing

    inline rat (a b)
        - ((a + b) / (a - b))
    let proj =
        mat4
            2.0 / (r - l)
            0
            0
            0

            0
            2.0 / (t - b)
            0
            0

            0
            0
            rat f n
            -1.0

            rat r l
            rat t b
            2 * (rat f n)
            0

    let translate =
        mat4
            vec4.Zero
            vec4.Zero
            vec4.Zero
            vec4 0 0 1 1

    proj * translate


struct MockElement
    image : Resource
    shader : (@ void)
    vertices : Resource
    pipeline : (@ void)
    flags : u64
    close : bool

fn behavior (element ctx back msg)
    global params =
        arrayof ShaderParamter
            ShaderParameter "MVP" 0 4 4 0 ShaderType.FLOAT
            ShaderParameter null  0 0 0 0 ShaderType.TEXTURE

    global counter : i32 = 0

    let element = (element as (@ MockElement))
