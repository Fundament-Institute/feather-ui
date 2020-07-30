{ pkgs ? import <nixpkgs> { }, ... }:

with pkgs;
with callPackage ./lib.nix { };

let
  backendInterface = stdenv.mkDerivation {
    name = "feather-backend-interface";
    version = "0.0.1";
    src = ./.;

    buildInputs = [ terra ];

    TERRA_PATH = "./?.t;deps/?.t";
    LUA_PATH = "./?.lua;deps/?.lua";

    dontConfigure = true;

    buildPhase = ''
      terra prebuild.t
    '';
    installPhase = ''
      mkdir -p $out/include
      cp backend.h $out/include/backend.h
    '';
  };
  withBackends = backends:
    stdenv.mkDerivation {
      name = "feather";
      version = "0.0.1";
      src = ./.;
      buildInputs = [ terra glibc.dev ];
      propagatedBuildInputs = backends;

      TERRA_PATH = "./?.t;deps/?.t";
      LUA_PATH = "./?.lua;deps/?.lua";

      installPhase = ''
        install -m 444 -Dt $out/share/terra/feather/ feather/*.t
        install -m 444 -Dt $out/share/terra/std std/*.t
      '';
      dontConfigure = true;
      dontBuild = true;

      checkPhase = ''
        mkdir bin-x64
        echo Testing phase 1
        terra tests/main.t
        echo
        echo Testing phase 2
        ./bin-x64/fgTests
      '';
      doCheck = true;

      passthru = { inherit withBackends backendInterface terra; };
    };
  # feather = stdenv.mkDerivation {
  #   name = "feather";
  #   version = "0.0.1";
  #   src = ./.;
  #   buildInputs = [ terra ];
  # };
  feather = withBackends [ (callPackage ./fgOpenGL { feather = feather; }) ];
  inherit terra;
in feather
