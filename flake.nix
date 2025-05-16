# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Fundament Software SPC <https://fundament.software>

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";

    crane.url = "github:ipetkov/crane";
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs =
    inputs@{ self
    , flake-utils
    , nixpkgs
    , rust-overlay
    , crane
    , advisory-db
    , ...
    }:
    flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ] (system:
    let
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };

      rust-custom-toolchain = (pkgs.rust-bin.stable.latest.default.override {
        extensions = [
          "rust-src"
          "rustfmt"
          "llvm-tools-preview"
          "rust-analyzer-preview"
        ];
      });
      impureDrivers = [ 
        "/run/opengl-driver" # impure deps on specific GPU, mesa, vulkan loader, radv, nvidia proprietary etc
      ];
      gfxDeps = [
        pkgs.xorg.libxcb
        pkgs.xorg.libX11
        pkgs.xorg.libXcursor
        pkgs.xorg.libXrandr
        pkgs.xorg.libXi
        pkgs.libxkbcommon
        pkgs.wayland
        pkgs.fontconfig # you probably need this? unless you're ignoring system font config entirely
        # pkgs.libGL/U # should be in /run/opengl-driver?
        pkgs.pkg-config # let things detect packages at build time
        # some toolkits use these for dialogs - probably not relevant for feather?
        # pkgs.kdialog
        # pkgs.yad
        pkgs.vulkan-loader
        #pkgs.libglvnd
      ];
    in
    rec {
      devShells.default =
        (pkgs.mkShell.override { stdenv = pkgs.llvmPackages.stdenv; }) {
          buildInputs = with pkgs; [ openssl pkg-config dotnet-sdk ] ++ gfxDeps;

          nativeBuildInputs = with pkgs; [
            # get current rust toolchain defaults (this includes clippy and rustfmt)
            rust-custom-toolchain

            cargo-edit
          ];

          #LD_LIBRARY_PATH = pkgs.lib.strings.concatMapStringsSep ":" toString (with pkgs; [ xorg.libX11 xorg.libXcursor xorg.libXi (libxkbcommon + "/lib") (vulkan-loader + "/lib") libglvnd ]);
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (impureDrivers ++ gfxDeps);
          # fetch with cli instead of native
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";
          RUST_BACKTRACE = 1;
        };

      checks =
        let
          craneLib =
            (inputs.crane.mkLib pkgs).overrideToolchain rust-custom-toolchain;
          commonArgs = {
            src = ./.;
            buildInputs = with pkgs; [ pkg-config openssl zlib ];
            strictDeps = true;
            version = "0.1.0";
            stdenv = pkgs: pkgs.stdenvAdapters.useMoldLinker pkgs.llvmPackages_15.stdenv;
            CARGO_BUILD_RUSTFLAGS = "-C linker=clang -C link-arg=-fuse-ld=${pkgs.mold}/bin/mold";
          };
          pname = "feather-checks";

          cargoArtifacts = craneLib.buildDepsOnly (commonArgs // {
            inherit pname;
          });
          build-tests = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts pname;
            cargoTestExtraArgs = "--no-run";
          });
        in
        {
          inherit build-tests;

          # Run clippy (and deny all warnings) on the crate source,
          # again, reusing the dependency artifacts from above.
          #
          # Note that this is done as a separate derivation so that
          # we can block the CI if there are issues here, but not
          # prevent downstream consumers from building our crate by itself.
          feather-clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            pname = "${pname}-clippy";
            cargoClippyExtraArgs = "-- --deny warnings";
          });

          # Check formatting
          feather-fmt = craneLib.cargoFmt (commonArgs // {
            pname = "${pname}-fmt";
          });

          # Audit dependencies
          feather-audit = craneLib.cargoAudit (commonArgs // {
            pname = "${pname}-audit";
            advisory-db = inputs.advisory-db;
            cargoAuditExtraArgs = "--ignore RUSTSEC-2020-0071";
          });

          # We can't run tests during nix flake check because it might not have a graphical device.
        };
    });
}
