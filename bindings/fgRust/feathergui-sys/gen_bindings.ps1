bindgen ..\..\..\include\fgAll.h -o src/lib.rs --whitelist-var "[^_].*" --whitelist-type "
.*" --whitelist-function "[^_].*" --raw-line "#![feature(untagged_unions)]" --raw-line "#![allow(non_camel_case_types, non_snake_case)]" -- -x c++ -std=c++1
4 -D_RUST_BINDGEN