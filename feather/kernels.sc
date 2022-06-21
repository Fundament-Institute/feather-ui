using import struct
import itertools

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
    'reverse collected

fn make-type (varyings uniforms func)

sugar kernel (name args...)
    let varyings uniforms body =
        sugar-match args...
        case (('curly-list uniform...) (varying...) body...)
            _ varying... uniform... body...
        case ((varying...) ('curly-list uniform...) body...)
            _ varying... uniform... body...
        case ((varying...) body...)
            _ varying... '() body...

    let vtypes = (parse-typed-arg-list varyings)
    let utypes = (parse-typed-arg-list uniforms)



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

#TODO slice



type executor
    spice map (self kernel args...)
        error "map is unimplemented on this executor"

type processor-executor < executor
    spice map (self kernel args...)
        let kernel-type = ('typeof kernel)
        let n-uniforms = (countof kernel-type.uniforms)
        let arg-uniforms = (lslice args... n-uniforms)
        let arg-varyings = (rslice args... n-uniforms)
        let output-type = (buffer kernel-type.return-type)
        spice-quote
            let sizes... = (va-map countof arg-varyings...)
            let result-size = (min sizes...)
            let output = (output-type result-size)
            for i in (range result-size)
                let arg-elems... = (va-map (inline (buff) buff @ i) arg-varyings...)
                (output @ i) = (kernel arg-uniforms arg-elems)
            output
