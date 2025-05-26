// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

use std::path::PathBuf;

fn get_cargo_target_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let profile = std::env::var("PROFILE")?;
    let mut target_dir = None;
    let mut sub_path = out_dir.as_path();
    while let Some(parent) = sub_path.parent() {
        if parent.ends_with(&profile) {
            target_dir = Some(parent);
            break;
        }
        sub_path = parent;
    }
    let target_dir = target_dir.ok_or("not found")?;
    Ok(target_dir.to_path_buf())
}

fn main() {
    uniffi::generate_scaffolding("src/calculator.udl").unwrap();

    // Attempt to build C# example and copy it to target dir
    match std::process::Command::new("dotnet")
        .args(["build", "calculator-cs/calculator-cs.csproj"])
        .spawn()
        .and_then(|mut c| c.wait())
        .and_then(|e| Ok(e.success()))
    {
        Ok(true) => {
            if let Ok(s) = get_cargo_target_dir() {
                let curdir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
                let bin = curdir.join("calculator-cs/bin");

                let debug = bin.read_dir().unwrap().last().unwrap().unwrap();
                let net8 = debug.path().read_dir().unwrap().last().unwrap().unwrap();

                std::fs::copy(
                    net8.path().join("calculator-cs.runtimeconfig.json"),
                    s.join("calculator-cs.runtimeconfig.json"),
                )
                .unwrap();
                std::fs::copy(
                    net8.path().join("calculator-cs.dll"),
                    s.join("calculator-cs.dll"),
                )
                .unwrap();
                if std::fs::copy(
                    net8.path().join("calculator-cs.exe"),
                    s.join("calculator-cs.exe"),
                )
                .is_err()
                {
                    std::fs::copy(net8.path().join("calculator-cs"), s.join("calculator-cs"))
                        .unwrap();
                }
            } else {
                print!("Couldn't get TARGET_DIR for current crate, C# example not copied to output dir.");
            }
        }
        // We do not panic on error here so systems without dotnet can still build the rust example
        Ok(false) => print!("dotnet build failed, calculator-cs will not be available."),
        Err(e) => print!(
            "Error running dotnet build, calculator-cs will not be available: {}",
            e
        ),
    }
}
