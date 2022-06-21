

let sail =
    include
        """"
            #include <stddef.h>
            #include <sail/sail.h>

load-library "libsail.so"
load-library "libsail-common.so"

# print "functions"
# for x in ('all sail.extern)
#     print x
# print;
# print "structs"
# for x in ('all sail.struct)
#     print x

run-stage; # needed to ensure libraries load before starting to use them

# print (typeof sail.extern.sail_load_from_file)


do
    type image : (mutable@ sail.struct.sail_image)
        inline... __typecall
        case (cls)
            local img : (mutable@ sail.struct.sail_image)
            sail.extern.sail_alloc_image &img
            bitcast img cls
        case (cls, width : u32, height : u32, bpp : u8)
            let img = (this-type)
            local buff : (@ void)
            let res =
                sail.extern.sail_malloc (* (width as u64) height bpp) &buff
            if (res != sail.const.SAIL_OK)
                raise res
            else
                img.pixels = buff
                img.width = width
                img.height = height
                img.bytes_per_line = (width * bpp)
                return img
        # case (cls, img : (mutable@ sail.struct.sail_image))
        #     bitcast img cls

        inline __drop (self)
            # print "dropping an image"
            sail.extern.sail_destroy_image (storagecast self)
            # print "dropped image"
        inline __getattr (self attr)
            getattr (storagecast self) attr

    fn load-from-file (path)
        # returning image
        raising sail.enum.SailStatus

        local img : (mutable@ sail.struct.sail_image)
        let res =
            sail.extern.sail_load_from_file path (& img)
        if (res != sail.const.SAIL_OK)
            raise res
        else
            bitcast img image

    fn load-from-memory (buff)
        returning image
        raising sail.enum.SailStatus

        local img : (mutable@ sail.struct.sail_image)
        let res =
            sail.extern.sail_load_from_memory buff (& img)
        if (res != sail.const.SAIL_OK)
            raise res
        else
            bitcast img image

    fn save-into-file (path img)
        raising sail.enum.SailStatus

        let res =
            sail.extern.sail_save_into_file path (storagecast img)
        if (res != sail.const.SAIL_OK)
            raise res

    # fn save-into-memory (buff img)
        raising sail.enum.SailStatus

        let res =
            sail.extern.sail_save_into_file path (storagecast img)
        if (res != sail.const.SAIL_OK)
            raise res

    type+ sail.enum.SailPixelFormat
        inline __repr (self)
            ..
                sail.extern.sail_pixel_format_to_string self
                ":"
                repr (typeof self)

    let raw = sail
    # load-from-memory :=

    locals;
