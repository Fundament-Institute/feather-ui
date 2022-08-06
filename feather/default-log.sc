

using import .backend
using import C.stdio

fn default-log (backend level file line msg values n-values free-fn)
    printf "%s [%s:%i] %s\n" (tostring level) file line msg
