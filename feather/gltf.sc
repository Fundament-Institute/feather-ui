
using import .json
using import itertools
using import Array
using import String
using import glm
import C.stdio


fn load-glb-from-path (path)
    let stream =
        C.stdio.fopen path "r"
    local header = ((array u32 3))
    C.stdio.fread (header as (mutable@ u32)) 4 3 stream
    print (header @ 0) (header @ 1) (header @ 2) # load bearing print
    if ((header @ 0) != 0x46546c67)
        error "file does not start with the right magic number"
    if ((header @ 1) != 2)
        error "file contains unsupported gltf version"
    let buffsize = ((header @ 3) - 12)
    local buff = (malloc-array u8 buffsize)
    C.stdio.fread buff 1 buffsize stream
    inline offset-ptr (ptr offset)
        inttoptr
            +
                ptrtoint ptr intptr
                offset
            (typeof ptr)
    inline read-u32 (buff offset)
        @
            inttoptr
                +
                    ptrtoint buff intptr
                    offset
                (@ u32)
    local chunks = ((GrowingArray (tuple (@ void) u32 u32)))
    loop (offset = 0:u32)
        if (offset >= buffsize)
            break;
        let chunksize = (read-u32 buff offset)
        'emplace-append chunks (offset-ptr buff (offset + 8)) (read-u32 buff (offset + 4)) chunksize
        repeat (offset + chunksize + 8)
    for chunk in chunks
        let base kind len = (unpack chunk)
        if (kind == 0x4e4f534a) # JSON chunk
            let str = (String (base as (@ i8)) len)
            print str





locals;
