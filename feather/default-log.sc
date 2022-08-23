using import .backend
using import C.stdio

global Levels =
    arrayof rawstring
        "Fatal: "
        "Error: "
        "Warning: "
        "Notice: "
        "Debug: "

        #FIXME no way to write a log function here.
        
fn default-log (backend level file line msg values n-values free-fn)
    printf "%s [%s:%i] %s\n" (Levels @ level) file line msg
    _;

locals;
