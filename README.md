# Feather UI

Feather is a universal UI library that applies user inputs to application state, and maps application state to an interactive visualization using a custom graphics rendering language capable of compiling to arbitrary GPU code or vectorized CPU code.

This project is currently in a prototyping stage, it is not yet suitable for production, and not all planned features are implemented.

## Building

Feather is a standard rust project, simply run `cargo build` on your platform of choice. A NixOS flake is included that provides a develop environment for nix developers who do not have rust installed system-wide.

## Running

Two working examples are available: `basic-rs` and `basic-alc`. To run either, navigate into the folder and run `cargo run`. Please note that `basic-alc` currently relies on an alicorn interpreter and is thus extremely slow, it will take about 30 seconds for it to run on a powerful system.

The examples have currently only been tested on NixOS and Windows 11, but should work on most systems.