

using import .backend
using import glm
using import struct
using import Option

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
    #image : Resource
    #shader : (@ void)
    #vertices : Resource
    #pipeline : (@ void)
    #flags : u64
    close : bool

let backend_new =
    extern 'fgOpenGL InitBackend

load-library "libfgOpenGL.so"

run-stage;

fn behavior (element ctx back msg)
    #global params =
        arrayof ShaderParameter
            ShaderParameter "MVP" 0 4 4 0 Shader_Type.FLOAT
            ShaderParameter null  0 0 0 0 Shader_Type.TEXTURE

    global counter : i32 = 0

    let element = (element as (@ MockElement))

    local res = (Msg_Result)
    res.draw = 0
    res

let behavior =
    static-typify
        behavior
        (@ void)
        (@ void)
        (@ void)
        (mutable@ Msg)


fn main ()
    using import .default-log
    let b = (backend_new null default-log behavior )
    local element : MockElement
    let w = ('create-window b &element none none (vec2 800 600) "smoke test" 0)
    while true
        'process-messages b w
    'destroy b w
    _;

main;
