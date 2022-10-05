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
            let state-type = (PromiseState T Err)
            inline then (self handler)
                let handler-prime = ('instance handler T)
                let handler-type = (typeof handler-prime)
                let ft = handler-type.FunctionType
                let res err = ('return-type ft)
                let collapsed-res promise-type collapse? =
                    static-if (type< res promise-base)
                        _ res.success-type res true
                    else
                        _
                            res
                            static-if (!= err Nothing)
                                promise-base res err
                            else
                                promise-base res Err
                            false


                let next-state-type = (PromiseState collapsed-res err)
                let intermediate-result-type = (PromiseResult res err)
                let next-result-type = (PromiseResult collapsed-res err)
                let state = (storagecast self)
                local next-state = (Rc.new (getattr next-state-type 'unbound))
                'bind-handler state
                    capture "promise-handler" {handler next-state}(result)
                        dispatch result
                        case ok (val)
                            let result =
                                try
                                    let ret = (handler val)
                                    let collapsed-ret =
                                        static-if collapse?
                                            'bind-handler (storagecast ret)
                                                capture {next-state}(val)
                                                    'recieve-result next-state val
                                        else
                                            'recieve-result next-state (next-result-type.ok ret)
                                except (ex)
                                    'recieve-result next-state (next-result-type.err ex)

                            case err (val)
                                'recieve-result next-state (next-result-type.err val)
                drop self

                bitcast next-state promise-type

            inline catch (self handler)

            inline cancel (self)
                let state = (storagecast self)

            fn wait (self)
                let el = thread-local-event-loop
                # TODO: add support for waiting on an event provider
                vvv bind res
                loop ()
                    if (not ('runnable? el))
                        break (error "tried to wait for a promise but ran out of events before it was resolved")
                    'run el 1000
                    let state = (storagecast self)
                    dispatch state
                    case unbound()
                        continue
                    case recieved (res)
                        state = this-type.state-type.finished
                        break res
                    default
                        break (error "promise in invalid state. did something else wait on a linear promise?")

                dispatch res
                case ok(val)
                    val
                case err(val)
                    raise val




locals;
