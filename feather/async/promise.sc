""""An implementation of promises in scopes

using import struct
using import enum
using import capture
using import .eventloop
using import Rc

@@ type-factory
inline PromiseResult(T Err)
    enum (.. "(PromiseResult " (tostring T) " " (tostring Err) ")")
        ok : T
        err : Err

@@ type-factory
inline PromiseState(T Err)
    enum (.. "(PromiseState " (tostring T) " " (tostring Err) ")")
        unbound
        bound : (Capture (Capture.function void (PromiseResult T Err)))
        recieved : (PromiseResult T Err)
        finished

        inline bind-handler (self handler)
            self =
                dispatch self
                case unbound ()
                    this-type.bound handler
                case recieved (result)
                    handler result
                    this-type.finished;
                default
                    error "tried to bind a handler in an invalid state"

        inline recieve-result (self result)
            self =
                dispatch self
                case unbound ()
                    this-type.recieved result
                case bound (handler)
                    handler result
                    this-type.finished;
                default
                    error "tried to recieve a result in an invalid state"

# any custom error type being handed to promises will need
type CancelError : void
    inline __imply (cls T)
        static-if (T == Error)
            inline "Cancel-as" (self)
                Error "Promise Canceled"
        else
            error "any custom error type being used in promises will need an __rimply implementation from CancelError"

type PromiseFulfiller
    inline rejectIfError(self func args...)
        """"Run a function such that if it errors, the promise this fulfiller handles will be rejected with the error.
            Returns true on success, false on error, and does not propogate errors.
        try
            func args...
            true
        except (ex)
            'reject self ex
            false

    inline make-typed-fulfiller (T Err)
        let ST = (tuple (@ EventLoop))
        type (.. "(PromiseFulfiller " (tostring T) " " (tostring Err) ")") < PromiseFulfiller :: ST
            fn... fulfill(self : this-type, val : T)
                """"Fulfill a promise with the given value
                let


inline newAdaptedPromise(Adapter args...)

type Promise
    let promise-base = this-type
    @@ type-factory
    inline make-type (T Err)
        type (.. "(Promise " (tostring T) " " (tostring Err) ")") :: (Rc (PromiseState T Err)) < this-type
            let success-type = T
            let fail-type = Err
            let result-type = (PromiseResult T Err)
            inline then (self handler)
                let handler-prime = ('instance handler T)
                let handler-type = (typeof handler-prime)
                let ft = handler-type.FunctionType
                let res err = ('return-type ft)
                let collapsed-res =
                    static-if (type< res promise-base)


                let next-state-type = (PromiseState res err)
                let next-result-type = (PromiseResult res err)
                let state = (storagecast self)
                local next-state = (Rc.new (getattr next-state-type 'unbound))
                'bind-handler state
                    capture "promise-handler" {handler next-state}(result)
                        dispatch result
                        case ok (val)
                            let result =
                                try
                                    let ret = (handler val)
                            let ret = (accept val)
                            let ret-type = (typeof ret)
                            static-if (type< res promise-base)
                                ret
                            else

                        case err (val)
                            reject val

                drop self

            inline catch (self handler)

            inline cancel (self)
                let state = (storagecast self)



locals;
