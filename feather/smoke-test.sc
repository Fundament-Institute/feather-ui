using import .backend
using import glm
using import struct
using import Option
# using import .executors.opengl

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


struct MockElement plain
    #image : Resource
    #shader : (@ void)
    #vertices : Resource
    #pipeline : (@ void)
    #flags : u64
    close : bool

let backend_new =
    load-backend 'fgOpenGL MockElement

static-if (operating-system == 'windows) 
    load-library "bin/fgOpenGL.dll"
else 
    load-library "libfgOpenGL.so"

run-stage;

fn behavior (ctx msg app id)
    #global params =
        arrayof ShaderParameter
            ShaderParameter "MVP" 0 4 4 0 Shader_Type.FLOAT
            ShaderParameter null  0 0 0 0 Shader_Type.TEXTURE

    global counter : i32 = 0

    # let element = (element as (@ MockElement))
    dump app
    local app = app

    if (msg.kind == Event_Kind.KeyDown)
        print "got keydown" msg.keyDown.scancode
        app.close = true

    local res = (Msg_Result)
    res.draw = 0
    _ app res

fn main ()
    using import .default-log
    local logger : default-logger
    let b = (backend_new (& logger) behavior )
    local element : MockElement
    let w = ('create-window b 0 none none (vec2 800 600) "smoke test" 0)
    let context = ('get-context w)
    # let executor = (opengl-executor b context)
    while (not element.close)
        element = ('process-messages b element w)
    'destroy b w
    _;

main;
