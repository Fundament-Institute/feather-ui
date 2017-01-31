use std::env;
use std::path::PathBuf;

#[cfg(windows)]
static LIB_FILE: &'static str = "feathergui.lib";
#[cfg(not(windows))]
static LIB_FILE: &'static str = "libfeathergui.a";

#[cfg(target_pointer_width = "32")]
static BIN_DIR: &'static str = "bin32";
#[cfg(target_pointer_width = "64")]
static BIN_DIR: &'static str = "bin";

fn get_lib_dir() -> PathBuf {
    let root = PathBuf::from(env::var("FEATHERGUI_DIR").expect("FEATHERGUI_DIR must be set"));
    let bin_dir = root.join(BIN_DIR);
    let lib_file = bin_dir.join(LIB_FILE);

    if !lib_file.exists() {
        panic!("feathergui library does not exist, did you build it?");
    }

    bin_dir
}

fn main() {
    println!("cargo:rustc-link-search=native={}", get_lib_dir().to_str().unwrap());
    println!("cargo:rustc-link-lib=static=feathergui");
}
