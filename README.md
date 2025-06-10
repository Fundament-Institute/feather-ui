# Feather UI

Feather is a universal UI library that applies user inputs to application state, and maps application state to an interactive visualization using a custom graphics rendering language capable of compiling to arbitrary GPU code or vectorized CPU code.

This project is currently in a prototyping stage, it is not yet suitable for production, and not all planned features are implemented.

## Building

Feather is a standard rust project, simply run `cargo build` on your platform of choice. A NixOS flake is included that provides a develop environment for nix developers who do not have rust installed system-wide.

## Running

Two working examples are available: `basic-rs` and `paragraph-rs`. To run either, navigate into the folder and run `cargo run`.

If you are on NixOS, use `nix run github:Fundament-Software/feathergui#basic-rs` or `nix run github:Fundament-Software/feathergui#paragraph-rs`  
If you are not on nixos but have nix, use `nix run --impure github:nix-community/nixGL -- nix run github:fundament-software/feathergui#basic-rs`  
If you do not have nix, you must navigate into the relevant example folder and run `cargo run`

The examples have currently only been tested on NixOS and Windows 11, but should work on most systems.

## Funding

This project is funded through [NGI Zero Core](https://nlnet.nl/core), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. Learn more at the [NLnet project page](https://nlnet.nl/project/FeatherUI).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)
[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

## License
Copyright © 2025 Fundament Software SPC

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: Apache-2.0
