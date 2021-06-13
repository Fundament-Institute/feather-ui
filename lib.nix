{ pkgs }:

with pkgs.lib;

let
  terra = pkgs.callPackage (pkgs.fetchFromGitHub {
    owner = "Fundament-Software";
    repo = "terra";
    rev = "fb410b202e26c58003dcad4138d42f1189d5954c";
    sha256 = "1x0f2k5wm1vpazg7v57r7bp5hyfsp5075az4j00qn06y5wyq7g8a";
  }) { };
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
