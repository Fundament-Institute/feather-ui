using import struct
using import itertools

fn parse-typed-arg-list (args)
    let collected ignore =
        loop (collected rest = '() args)
            sugar-match rest
            case ((name as Symbol) ': typ ', rest...)
                repeat
                    cons
                        tupleof name typ
                        collected
                    rest...
            case ((name as Symbol) ', rest...)
                repeat
                    cons
                        tupleof name Unknown
                        collected
                    rest...
            case ((name as Symbol) ': typ)
                break
                    cons
                        tupleof name typ
                        collected
                    '()
            case ((name as Symbol))
                break
                    cons
                        tupleof name Unknown
                        collected
                    '()
            default
                error "unknown argument syntax in typed arg list"
    'reverse collected

fn make-kernel-type (varyings uniforms func)
    type kernel
        let uniforms
        let varyings
        let static-func = static-typify

""""
    kernels may be written as
    kernel foo (v1 : t1, v2 : t2) {u1 : t3}
        body...
    where v1 and v2 are varying parameters and u1 is a uniform parameter.
    The body may have any ordinary scopes code as a function in it.
#sugar kernel (name args...)
    let varyings uniforms body =
        sugar-match args...
        case (('curly-list uniform...) (varying...) body...)
            _ varying... uniform... body...
        case ((varying...) ('curly-list uniform...) body...)
            _ varying... uniform... body...
        case ((varying...) body...)
            _ varying... '() body...
        default
            error "invalid kernel syntax, please fix this error message to be useful"

    let vtypes = (parse-typed-arg-list varyings)
    let utypes = (parse-typed-arg-list uniforms)
    inline names-of (params)
        vvv 'reverse
        ->> params
            map
                inline (tup)
                    tup @ 0
            'cons-sink '()
    let vnames = (names-of vtypes)
    let unames = (names-of utypes)
    let func =
        qq
            fn ((unquote-splice unames) (unquote-splice vnames))
                (unquote-splice body)


type buffer
    @@ memo
    inline make-type (element)
        let ST = (tuple (pointer element) usize)
        type (.. "buffer<" (tostring element) ">") < this-type :: ST
            let element

    inline __typecall (cls args...)
        static-if (cls == this-type)
            make-type args...
        else
            let count initializer = args...
            let buff = (malloc-array cls.element count)
            for i in (range count)
                (@ buff i) = (initializer i)
            bitcast (tupleof buff count) cls

    inline __@ (self idx)
        idx as:= usize
        let base size = (unpack (storagecast self))
        assert (idx < size) "index out of bounds"
        base @ idx

    inline __drop (self)
        returning void
        let base size = (unpack (storagecast self))
        for i in (range size)
            __drop (base @ i)
        free base
        _;

#TODO slice


@@ type-factory
inline foreign-buffer (elem executor)
    let ST = (tuple (@ void) usize (& executor))
    type (.. "(foreign-buffer" (tostring elem) ")") : ST
        inline __drop (self)
            let id size exec = (unpack (storagecast self))
            ('delete-buffer exec id)

        inline elem-type (self)
            elem

type executor
    spice map (self kernel args...)
        returning Value
        error "map is unimplemented on this executor"
    spice upload-buffer (self buff)
        returning Value
        error
            ..
                "upload-buffer must be implemented by executors but is not implemented by "
                tostring ('typeof self)
    spice download-buffer (self fbuff)
        returning Value
        error
            ..
                "download-buffer must be implemented by executors but is not implemented by "
                tostring ('typeof self)
    spice new-buffer (self typ size initializer)
        returning Value
        error
            ..
                "new-buffer must be implemented by executors but is not implemented by "
                tostring ('typeof self)

type rendering-executor < executor
    spice upload-texture (self buff)
        returning Value
        error
            ..
                "upload-texture must be implemented by rendering-executors but is not implemented by "
                tostring ('typeof self)
    spice download-texture (self fbuff)
        returning Value
        error
            ..
                "download-texture must be implemented by rendering-executors but is not implemented by "
                tostring ('typeof self)
    spice new-texture (self typ size initializer)
        returning Value
        error
            ..
                "new-texture must be implemented by rendering-executors but is not implemented by "
                tostring ('typeof self)

    spice run-pass(self operation)
        returning Value
        error
            ..
                "run-pass must be implemented by rendering-executors but is not implemented by "
                tostring ('typeof self)

    spice create-pipeline (self config...)
        returning Value
        error
            ..
                "create-pipeline must be implemented by rendering-executors but is not implemented by "
                tostring ('typeof self)




#type processor-executor < executor
    spice upload-buffer (self buff)
        buff
    spice download-buffer (self buff)
        buff
    inline new-buffer (self typ size initializer)
        ((buffer typ) size initializer)
    spice map (self kernel args...)
        let kernel-type = ('typeof kernel)
        let n-uniforms = (countof kernel-type.uniforms)
        let arg-uniforms... = (lslice args... n-uniforms)
        let arg-varyings... = (rslice args... n-uniforms)
        let output-type = (buffer kernel-type.return-type)
        spice-quote
            let sizes... = (va-map countof arg-varyings...)
            let result-size = (min sizes...)
            let output =
                output-type result-size
                    inline (i)
                        let arg-elems... = (va-map (inline (buff) buff @ i) arg-varyings...)
                        kernel arg-uniforms... arg-elems...
            output


# Executors:
    single threaded CPU executor
    threadpool CPU executor
    CUDA executor (low priority)
    OpenGL executor
    Vulkan executor
    shader-of-compute executor
    feather backend executor?

# rendering executor
    rendering executor has methods to upload/download textures and buffers
    depth buffers and stencil buffers aggregated into a renderpass object
    renderpass objects may have multiple render target textures
    render method on an executor that takes the executor and a function that operates on a renderpass.
        the renderpass can't outlive the rendering function.
    vertex shaders are represented as a kernel taking any primitive varyings, and any combination of primitive and device-buffer uniforms
    vertex shader outputs are arbitrary device primitives, but must contain a position vec4 suitable for interpolation for use in the depth-test.
    pixel shaders are represented as a kernel taking any primitive varyings and any combination of primitive and device-buffer uniforms
    pixel shader varyings must match the vertex shader outputs
    pixel shader outputs must match the renderpass output elements
    vertex shader and pixel shader kernels are grouped into a pipeline object which tests the type matching.
    pipelines may be invoked on device data in a renderpass


#run-stage;
#TESTS
#fn test()
    let ex = (processor-executor)
    let xs =
        'new-buffer ex f32 128
            inline (i)
                i * 2
    let ys =
        'new-buffer ex f32 128
            inline (i)
                i * 3
    kernel saxpy (x : f32, y : f32) {a : f32}
        a * x + y
    let res =
        'map ex saxpy 0.5 xs ys
    'download-buffer ex res

locals;
