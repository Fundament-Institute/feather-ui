{
  description = "A minimal scopes project";
  inputs = {
    scopes.url = "github:Fundament-software/scopes";
    nixpkgs.follows = "scopes/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    sail-src.url = "github:HappySeaFox/sail";
    sail-src.flake = false;
  };

  outputs = { self, scopes, nixpkgs, flake-utils, nix-filter, sail-src }:
    # Using same set of systems as scopes.packages as we can't support systems
    # that scopes doesn't support
    (flake-utils.lib.eachSystem (builtins.attrNames scopes.packages) (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        selfpkgs = self.packages.${system};
        featherNativeDevDeps = [
          pkgs.pkg-config
          scopes.packages.${system}.scopes
        ];
        featherDevDeps = [
          backends
          pkgs.cjson
          selfpkgs.sail
        ];
        devshell-ldpath = pkgs.lib.makeLibraryPath [
          selfpkgs.sail
          selfpkgs.fgOpenGL
          pkgs.cjson
        ];
        # TODO: investigate something in scopes flake to automate or abstract this
        featherDevSetupHook = ''
          export NIX_CFLAGS_COMPILE="''${NIX_CFLAGS_COMPILE:-} $(pkg-config --cflags libcjson libsail)"
          export LD_LIBRARY_PATH=${devshell-ldpath}:''${LD_LIBRARY_PATH:-}
        '';
        backends = pkgs.llvmPackages_13.stdenv.mkDerivation {
          name = "backends";
          src = nix-filter.lib.filter {
            root = ./.;
            include = [
              "CMakeLists.txt"
              (nix-filter.lib.inDirectory "backendtest")
              (nix-filter.lib.inDirectory "fgOpenGL")
              (nix-filter.lib.inDirectory "fgGLFW")
              (nix-filter.lib.inDirectory "fgOpenGLDesktopBridge")
              (nix-filter.lib.inDirectory "include")
            ];
          };

          nativeBuildInputs = [ pkgs.cmake ];
          buildInputs = [ pkgs.libglvnd pkgs.glfw pkgs.xorg.libX11 pkgs.xorg.libXrandr ];
          outputs = [ "out" ];

          cmakeFlags = [ "-DUSE_DEFAULT_FOLDERS=1" ];

          # installPhase = ''
          #   mkdir -p $out/build-dump
          #   cp -r . $out/build-dump
          # '';
        };
        runCommandWithFeather = name: script:
          pkgs.runCommand name
            {
              nativeBuildInputs = featherNativeDevDeps;
              buildInputs = featherDevDeps;
            } ''
            set -euo pipefail
            ln -s "${./feather}" feather
            mkdir output
            ${featherDevSetupHook}
            # scopes needs writable $HOME for ~/.cache/scopes
            export HOME=$TMP
            ${script}
            ls output
            mv output $out
          '';
      in {
        packages = {
          fgOpenGL = backends;
          sail = pkgs.stdenv.mkDerivation {
            name = "sail";
            src = sail-src;

            cmakeFlags = [ "-DSAIL_COMBINE_CODECS=ON" ];
            buildInputs =
              [ pkgs.cmake pkgs.libpng pkgs.libjpeg_turbo pkgs.libwebp ];
          };
        };
        devShell = pkgs.mkShell {
          nativeBuildInputs = featherNativeDevDeps;
          buildInputs = featherDevDeps;

          shellHook = featherDevSetupHook;
        };
        checks = {
          feather-tests = runCommandWithFeather "feather-tests" ''
            for test in feather/{json,rendering}-test.sc; do
              echo Running $test
              scopes $test
            done
          '';
        };
      })) // {

      };

}
