# Feather UI

Feather is a universal UI library that only mutates application state in response to user inputs or events, using functional reactive event streams, and maps application state to a layout using persistent functions, which then efficiently render only the parts of the UI that changed using either a standard GPU compositor or custom shaders.

## Building

Feather is a standard rust project, simply run `cargo build` on your platform of choice. A NixOS flake is included that provides a develop environment for nix developers who do not have rust installed system-wide.

## Running

Examples can be found in `feather-ui/examples`, and can be run via `cargo run --example [example_name]`. 

If you are on NixOS, use `nix run github:Fundament-Software/feathergui#[example_name]`
If you are not on nixos but have nix, use `nix run --impure github:nix-community/nixGL -- nix run github:fundament-software/feathergui#[example_name]`  

The examples have currently only been tested on NixOS and Windows 11, but should work on most systems.

## Funding

This project is funded through [NGI Zero Core](https://nlnet.nl/core), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. Learn more at the [NLnet project page](https://nlnet.nl/project/FeatherUI).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)
[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

## License
Copyright © 2025 Fundament Software SPC

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: Apache-2.0
