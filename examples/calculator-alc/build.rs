// SPDX-License-Identifier: MPL-2.0

fn main() {
  uniffi::generate_scaffolding("src/calculator.udl").unwrap();
  uniffi_alicorn::generate_alicorn_scaffolding("src/calculator.udl").unwrap();
}
