
using import struct
using import Array
using import Option
using import Rc
using import itertools
using import glm


let raw =
    include
        """"#include "feather/shared_interface.h"
            #include "feather/desktop_interface.h"
            #include "feather/graphics_interface.h"
            #include "feather/graphics_desktop_bridge.h"
            #include "stdarg.h"
            #include "stdio.h"



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

spice bitfield_gen (cls vals...)
    va-lfold
        inline (a b)
            `( a or b )
        va-map
            inline (v)
                `(getattr cls (v as Symbol))
            vals...


run-stage;

sugar wrap-enum (name)
    name as:= Symbol
    let expandedName = ("FG_" .. (name as string))
    qq
        [let] [name] =
            [type+] ([getattr] [raw.enum] (sugar-quote [(Symbol expandedName)]))
                [enum-vals] [(expandedName .. "_")]

sugar wrap-bitfield (name)
    name as:= Symbol
    let expandedName = ("FG_" .. (name as string))
    qq
        [let] [name] =
            [type+] ([getattr] [raw.enum] (sugar-quote [(Symbol expandedName)]))
                [enum-vals] [(expandedName .. "_")]
                let gen = [bitfield_gen]

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

sugar option-to-pointer (name val)
    let _name = (Symbol (.. "_" (name as Symbol as string)))
    qq
        local [_name] : (getattr (typeof [val]) 'elem)
        let [name] =
            dispatch [val]
            case Some (x)
                [_name] = x
                (&[_name])
            case None ()
                nullof (typeof (& [_name]))
            default
                unreachable;

run-stage;

inline unwrap-or (opt val)
    dispatch opt
    case Some (val)
        val
    case None ()
        val
    default
        unreachable;

do
    # Shared
    type+ raw.typedef.FG_Vec2
        inline __rimply (othercls cls)
            inline (other)
                cls
                    x = other.x
                    y = other.y
    let Vec2 = raw.typedef.FG_Vec2

    type+ raw.typedef.FG_Vec2i
        inline __rimply (othercls cls)
            inline (other)
                cls
                    x = (imply other.x i32)
                    y = (imply other.y i32)
    let Vec2i = raw.typedef.FG_Vec2i

    type+ raw.typedef.FG_Vec3
        inline __rimply (othercls cls)
            inline (other)
                cls
                    x = other.x
                    y = other.y
                    z = other.z
    let Vec3 = raw.typedef.FG_Vec3

    type+ raw.typedef.FG_Vec3i
        inline __rimply (othercls cls)
            inline (other)
                cls
                    x = (imply other.x i32)
                    y = (imply other.y i32)
                    z = (imply other.z i32)
    let Vec3i = raw.typedef.FG_Vec3i

    type+ raw.typedef.FG_Rect
        inline __rimply (othercls cls)
            inline (other)
                cls # expects cls to be a vec4
                    left = other.x
                    top = other.y
                    right = other.z
                    bottom = other.w
    let Rect = raw.typedef.FG_Rect

    type+ raw.typedef.FG_Color8
        inline __rimply (othercls cls)
            inline (other)
                cls
                    v = (imply other i32)
    let Color8 = raw.typedef.FG_Color8

    type+ raw.typedef.FG_Color16
        inline __rimply (othercls cls)
            inline (other)
                cls
                    v = (imply other i64)
    let Color16 = raw.typedef.FG_Color16

    wrap-enum Level
    wrap-enum LogType
    let LogValue = raw.typedef.FG_LogValue
    let Context = raw.typedef.FG_Context
    let Log = raw.typedef.FG_Log

    # Desktop
    let Display = raw.typedef.FG_Display
    wrap-enum Clipboard
    wrap-enum Cursor
    wrap-enum Event_Kind
    wrap-enum Keys
    wrap-bitfield ModKey # TODO make bitfield
    wrap-enum MouseButton
    wrap-enum JoyAxis
    wrap-enum Joy
    let Msg = raw.typedef.FG_Msg
    let Msg_Result = raw.typedef.FG_Result
    wrap-bitfield WindowFlag # TODO make bitfield
    let Behavior = raw.typedef.FG_Behavior

    type GenericDesktopInterface :: (mutable@ raw.struct.FG_DesktopInterface)

    @@ memo
    inline DesktopInterface (app-type)

        type (.. "DesktopInterface" (tostring app-type)) :: (mutable@ raw.struct.FG_DesktopInterface)
            inline __drop (self)
                let vtab = (storagecast self)
                vtab.destroy vtab

            let desktop-interface = this-type
            struct Window
                # parent : (& desktop-interface)
                window : (mutable@ raw.typedef.FG_Window)

                # inline __drop(self)
                    let vtab = (storagecast self.parent)
                    vtab.destroyWindow vtab self.window
                    _;
                fn get-context (self)
                    self.window.context
                spice __drop (self)
                    returning Value
                    error "window cannot be dropped, must be destroyed by desktop interface"


            unlet desktop-interface

            fn... create-window (self, id : u64, display : (Option Display), pos : (Option vec2), size : (Option vec2), caption, flags)
                viewing caption
                viewing self
                # returning (uniqueof Window 1003)

                # option-to-pointer disp display
                let disp = null
                # option-to-pointer pos pos
                let pos = null
                # option-to-pointer size size
                local _size : Vec2
                let psize =
                    dispatch size
                    case Some (x)
                        _size = x
                        &_size
                    case None ()
                        nullof (typeof (& _size))
                    default
                        unreachable;

                caption as:= rawstring

                let vtab = (storagecast self)
                let win =
                    vtab.createWindow vtab id disp pos psize caption flags

                Window
                    # parent = (view self)
                    window = win
                # _;

            fn process-messages (self app window)
                viewing self
                let vtab = (storagecast self)
                vtab.processMessages vtab window.window (& app)
                app

            fn... destroy
            case (self, w : Window)
                let vtab = (storagecast self)
                vtab.destroyWindow vtab w.window
                lose w

    let InitDesktopInterface = ((mutable@ raw.struct.FG_DesktopInterface) <-: ((@ void) Log Behavior))

    inline load-desktop-interface (name app-type)
        let raw-init =
            extern name InitDesktopInterface
        spice "desktop-interface-init" (logger handler)
            let logger-type = ('strip-pointer-storage-class ('typeof logger))
            spice-quote
                fn inner-handler (ctx msg app id)
                    returning Msg_Result
                    app as:= (mutable@ app-type)
                    let app-in = (app as (@ app-type))
                    let state res = (handler ctx msg (@ app-in) id)
                    dump state
                    (@ app) = state
                    res
                let inner-handler =
                    static-typify
                        inner-handler
                        (@ void)
                        (mutable@ Msg)
                        (@ void)
                        intptr
                fn inner-logger (self level file line text values n-values free-fn)
                    self as:= logger-type
                    values as:= (@ LogValue)
                    'log (@ self) level file line text values n-values free-fn
                    _;
                let inner-logger =
                    static-typify
                        inner-logger
                        (@ void)
                        Level
                        rawstring
                        i32
                        rawstring
                        (mutable@ LogValue)
                        i32
                        (@ (void <-: ((mutable rawstring))))
                bitcast
                    raw-init (logger as (@ void)) inner-logger inner-handler
                    DesktopInterface app-type

    # Graphics
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
    let GraphicsCaps = raw.typedef.FG_Caps
    wrap-enum Primitive
    wrap-enum Comparison
    wrap-enum Strip_Cut_Value
    wrap-enum Blend_Operand
    wrap-enum Blend_Op
    wrap-bitfield Pipeline_Member # TODO make bitfield
    wrap-enum Fill_Mode
    wrap-enum Cull_Mode
    wrap-bitfield Pipeline_Flags # TODO make bitfield
    wrap-enum Stencil_Op
    let PipelineState = raw.typedef.FG_PipelineState
    wrap-enum Filter
    wrap-enum Texture_Address_Mode
    let Sampler = raw.typedef.FG_Sampler
    wrap-enum Logic_Op
    let BlendState = raw.typedef.FG_Blend
    let Viewport = raw.typedef.FG_Viewport
    wrap-enum Vertex_Type
    let VertexParameter = raw.typedef.FG_VertexParameter
    wrap-enum Shader_Type
    let ShaderParameter = raw.typedef.FG_ShaderParameter
    let CommandList = raw.typedef.FG_CommandList
    let ShaderParameter = raw.typedef.FG_ShaderParameter
    let ShaderValue = raw.typedef.FG_ShaderValue
    wrap-bitfield AccessFlag
    wrap-bitfield BarrierFlags
    wrap-bitfield ClearFlag





    locals;
