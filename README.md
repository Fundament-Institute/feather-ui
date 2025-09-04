# Feather UI
<img align="right" width="20%" src="feather.svg">

[![Build Status](https://img.shields.io/github/actions/workflow/status/Fundament-Institute/feather-ui/ci.yml?branch=main&logo=github&label=CI)](https://github.com/Fundament-Institute/feather-ui/actions)
[![Discord](https://img.shields.io/static/v1?label=Channel&message=%23feather&color=blue&logo=discord)](https://discord.gg/mU7deD8DmT)

Feather is a reactive data-driven UI framework that only mutates application state in response to user inputs or events, using event streams and reactive properties, and represents application state using persistent data structures, which then efficiently render only the parts of the UI that changed using either a standard GPU compositor or custom shaders.

## Building

Feather is a standard rust project, simply run `cargo build` on your platform of choice. A NixOS flake is included that provides a develop environment for nix developers who do not have rust installed system-wide.

## Running

Examples can be found in [feather-ui/examples](feather-ui/examples), and can be run via `cargo run --example <example_name>`. 

If you are on NixOS, use `nix run github:Fundament-Software/feathergui#<example_name>`
If you are not on nixos but have nix, use `nix run --impure github:nix-community/nixGL -- nix run github:fundament-software/feathergui#<example_name>`  

The examples have currently only been tested on NixOS and Windows 11, but should work on most systems.

## Community

We have a [discord server](https://discord.gg/mU7deD8DmT) where you can ask questions and talk with contributors. However, if you have found an issue, you should prioritize [opening an issue](https://github.com/Fundament-Institute/feather-ui/issues/new) on GitHub rather than using Discord.

## Funding

This project is funded through [NGI Zero Core](https://nlnet.nl/core), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. Learn more at the [NLnet project page](https://nlnet.nl/project/FeatherUI).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)
[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

### Donations

Feather is developed by a non-profit, and we rely on community support to continue our work. Visit our [Open Collective](https://opencollective.com/feather-ui) if you are interested in donating towards a specific development goal, or just want to support us. We appreciate any and all help!

## License
Copyright © 2025 Fundament Research Institute

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: Apache-2.0
