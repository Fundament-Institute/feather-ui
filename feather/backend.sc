
using import struct
using import Array
using import Rc
using import itertools
using import glm


let raw =
    include
        """"#include "feather/backend.h"

            const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "DEBUG: " };

            // A logging function that just forwards everything to printf
            void FakeLog(void* root, FG_Level level, const char* f, ...)
            {

            if(level >= 0)
                printf("%s", LEVELS[level]);

            va_list args;
            va_start(args, f);
            vprintf(f, args);
            va_end(args);
            printf("\n");
            }

let enum-vals =
    sugar-scope-macro
        fn (args scope)
            source-scope := raw.const
            let prefix rest = (decons args)
            assert (rest == '())
            prefix as:= string
            let bindings =
                ->> source-scope
                    filter
                        inline (sym)
                            ==
                                lslice (sym as Symbol as string) (countof prefix)
                                prefix
                    map
                        inline (sym)
                            let basename =
                                rslice (sym as Symbol as string) (countof prefix)
                            _ (Symbol basename) (sym as Symbol)
                    map
                        inline (name orig)
                            tupleof name orig
                    'cons-sink '()
            let result-scope =
                fold (result-scope = scope) for names in bindings
                    let name binding = (unpack (names as (tuple Symbol Symbol)))
                    # print name binding
                    'bind result-scope name `(getattr source-scope binding)
            # print result-scope
            _ '() result-scope

spice wrap-fn-call (f args...)
    let ft = ('typeof f)
    let return-type = (sc_function_type_return_type ft)
    if ('pointer? return-type)
        spice-quote
            do
                let res = (f args...)
                if (res == null)
                    error (.. (spice-unquote (tostring f)) " returned a null pointer")
                else
                    res


run-stage;

sugar wrap-enum (name)
    name as:= Symbol
    let expandedName = ("FG_" .. (name as string))
    qq
        [let] [name] =
            [type+] ([getattr] [raw.enum] (sugar-quote [(Symbol expandedName)]))
                [enum-vals] [(expandedName .. "_")]

inline underscorify (prefix sym)
    let name = (sym as Symbol as string)
    let repl =
        ->> name
            map
                inline (c)
                    match c
                    case 45
                        95:i8
                    default
                        c
            string.collector (countof name)
    Symbol (prefix .. repl)

inline uppercase (sym)
    let name = (sym as Symbol as string)
    let repl =
        ->> name
            map
                inline (c)
                    if (c >= 97 & c < 122)
                        c - 32
                    else
                        c
            string.collector (countof name)
    Symbol repl


run-stage;

do
    wrap-enum PixelFormat
    wrap-enum Usage
    wrap-enum ShaderStage
    wrap-enum Feature
    let FeatureEnum = Feature
    unlet Feature
    type Feature : u32
        inline __getattr(self field)
            static-match field
            case 'api-opengl-es
                self & 7 == FeatureEnum.API_OpenGL_ES
            case 'api-opengl
                self & 7 == FeatureEnum.API_OpenGL
            case 'api-directx
                self & 7 == FeatureEnum.API_DirectX
            case 'api-vulkan
                self & 7 == FeatureEnum.API_Vulkan
            case 'api-metal
                self & 7 == FeatureEnum.API_Metal
            default
                # BUG: this is incorrect after refactor
                self & (getattr FeatureEnum (underscorify "" (uppercase field))) != 0
    wrap-enum Primitive
    wrap-enum Comparison
    wrap-enum Strip_Cut_Value
    wrap-enum Blend_Operand
    wrap-enum Blend_Op
    wrap-enum Pipeline_Member # TODO make bitfield
    wrap-enum Fill_Mode
    wrap-enum Cull_Mode
    wrap-enum Pipeline_Flags # TODO make bitfield
    wrap-enum Stencil_Op
    wrap-enum Filter
    wrap-enum Texture_Address_Mode
    wrap-enum Logic_Op
    wrap-enum Vertex_Type
    wrap-enum Shader_Type
    wrap-enum Clipboard
    wrap-enum Cursor
    wrap-enum Kind
    wrap-enum Keys
    wrap-enum ModKey # TODO make bitfield
    wrap-enum MouseButton
    wrap-enum JoyAxis
    wrap-enum Joy
    wrap-enum Level
    wrap-enum LogType
    wrap-enum WindowFlag # TODO make bitfield
    wrap-enum AccessFlags
    wrap-enum BarrierFlags
    wrap-enum ClearFlags

    let fake-log = raw.extern.FakeLog

    let ShaderParameter = raw.typedef.FG_ShaderParameter

    type Backend : raw.struct.FG_Backend

    locals;
