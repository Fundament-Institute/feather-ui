// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

fn main() {
    uniffi::generate_scaffolding("src/calculator.udl").unwrap();
}
