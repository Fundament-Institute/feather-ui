{
  description = "A minimal scopes project";
  inputs = {
    scopes.url = "github:Fundament-software/scopes";
    nixpkgs.follows = "scopes/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
  };

  outputs = { self, scopes, nixpkgs, flake-utils, nix-filter }:
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        backends = pkgs.llvmPackages_13.stdenv.mkDerivation {
          name = "backends";
          src = nix-filter.lib.filter {
            root = ./.;
            include = [
              "CMakeLists.txt"
              (nix-filter.lib.inDirectory "cpptest")
              (nix-filter.lib.inDirectory "fgOpenGL")
              (nix-filter.lib.inDirectory "include")
            ];
          };

          nativeBuildInputs = [ pkgs.cmake ];
          buildInputs = [ pkgs.libglvnd pkgs.glfw ];
          outputs = [ "out" ];

          cmakeFlags = [ "-DUSE_DEFAULT_FOLDERS=1" ];

          # installPhase = ''
          #   mkdir -p $out/build-dump
          #   cp -r . $out/build-dump
          # '';
        };
      in {
        packages = { fgOpenGL = backends; };
        devShell =
          pkgs.mkShell { buildInputs = [ scopes.packages.${system}.scopes ]; };
      })) // {

      };

}
