// SPDX-License-Identifier: MPL-2.0

fn main() {
    println!("cargo::rerun-if-changed=src/calculator.udl");
    println!("cargo::rerun-if-changed=layout.alc"); // avoid cachelighting when rust behaves badly and doesn't realize the file changed
    uniffi::generate_scaffolding("src/calculator.udl").unwrap();
    uniffi_alicorn::generate_alicorn_scaffolding("src/calculator.udl").unwrap();
}
