
using import struct
using import ..kernels
using import ..backend

struct opengl-data
    backend : &Backend
    context : &Context


type opengl-executor < rendering-executor : opengl-data

    fn __typecall (cls backend context)
        bitcast
            opengl-data
                backend = backend
                context = context
            cls

    inline upload-buffer (self buff)
        let data = (storagecast self)
        let glresult = ('create-buffer data.backend data.context ('raw-pointer buff) ('length buff) Usage.Storage_Buffer)
        if (glresult == null)
            error "something went wrong while creating buffer"
        (bitcast (tupleof glresult ('length buff) self) (foreign-buffer ('elem-type buff) this-type))

    inline download-buffer (self fbuff)
        let data = (storagecast self)
        let id size exec = (unpack (storagecast fbuff))
        let mapresult = ('map-resource data.backend data.context id 0 size Usage.Storage_Buffer AccessFlag.Read)
        let elem = ('elem-type fbuff)
        if (mapresult == null)
            error "something went wrong while mapping buffer"
        let buff =
            buffer
                size
                inline (i)
                    (mapresult as (@ elem)) @ i
        let unmapresult = ('unmap-resource data.backent data.context id Usage.Storage_Buffer)
        if (unmapresult != 0)
            error (.. "something went wrong while unmapping buffer. error code: " (tostring unmapresult))
        buff


locals;
