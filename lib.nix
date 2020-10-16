{ pkgs }:

with pkgs.lib;

let
  llvmPackages = pkgs.llvmPackages_9;
  luajitCommit = "9143e86498436892cb4316550be4d45b68a61224";
  luajitArchive = "LuaJIT-${luajitCommit}.tar.gz";
  luajitSrc = pkgs.fetchurl {
    url = "https://github.com/LuaJIT/LuaJIT/archive/${luajitCommit}.tar.gz";
    sha256 = "0kasmyk40ic4b9dwd4wixm0qk10l88ardrfimwmq36yc5dhnizmy";
  };
  terra = pkgs.terra.overrideAttrs (old: {
    src = pkgs.fetchFromGitHub {
      owner = "terralang";
      repo = "terra";
      rev = "4b94a993c2f5d38fe899e2cc7d3daa9c56af5567";
      sha256 = "11vplcdajh8bjl6rb8n7dz83ymdi4hsicsrp2jwg41298va744gx";
      # date = 2020-10-12T13:54:03-07:00;
    };

    buildInputs = with llvmPackages; [ llvm clang-unwrapped pkgs.ncurses ];
    meta = old.meta // { platforms = platforms.all; };

    preBuild = ''
      cat >Makefile.inc<<EOF
      CLANG = ${getBin llvmPackages.clang-unwrapped}/bin/clang
      LLVM_CONFIG = ${getBin llvmPackages.llvm}/bin/llvm-config
      EOF

      mkdir -p build
      cp ${luajitSrc} build/${luajitArchive}
      echo build/${luajitArchive}
    '';

    passthru = {

    };
  });
  SOIL = pkgs.stdenv.mkDerivation {
    name = "SOIL";
    version = "1.16";
    src = ./deps/SOIL;
    buildInputs = [ pkgs.cmake pkgs.libglvnd pkgs.xorg.libX11 ];
    postInstall = ''
      mv $out/include/SOIL/* $out/include/
    '';
  };
  freetype2 = pkgs.stdenv.mkDerivation {
    name = "Freetype2";
    version = "2.10.0";
    src = ./deps/freetype2;
    buildInputs = [ pkgs.cmake ];
  };
  harfbuzz = pkgs.stdenv.mkDerivation {
    name = "Harfbuzz";
    version = "2.6.8";
    src = ./deps/harfbuzz;
    buildInputs = [ pkgs.cmake freetype2 ];
  };
  glfw = pkgs.glfw3.overrideAttrs (old: { src = ./deps/glfw; });
  terraIncludes = lst:
    builtins.concatStringsSep ";" (map (pkg: "${getDev pkg}/include") lst);
  terraLibs = lst:
    builtins.concatStringsSep ";" (map (pkg: "${getLib pkg}/lib") lst);
  terraShell = { buildInputs, ... }@args:
    pkgs.mkShell (args // {
      INCLUDE_PATH = terraIncludes buildInputs;
      LIBRARY_PATH = terraLibs buildInputs;
      buildInputs = [ terra ] ++ buildInputs;
    });
in { inherit terraShell terra SOIL harfbuzz glfw freetype2; }
