{ config ? { }, lib ? { }, pkgs ? import <nixpkgs> { }
, feather ? pkgs.callPackage ../. { }, SOIL, harfbuzz, glfw, freetype2, ... }:

let inherit (pkgs) stdenv;
in stdenv.mkDerivation rec {
  name = "fgOpenGL";
  version = "0.0.1";
  makeFlags = [ "BINDIR=bin" "LIBDIR=lib" "OBJDIR=bin/obj" ];
  src = ./.;
  CPPFLAGS =
    "-I. -Wall -Wshadow -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch -std=c++17 -msse -msse2 -msse3 -mmmx -m3dnow -mcx16";

  buildInputs = [
    feather.backendInterface
    SOIL
    harfbuzz
    glfw
    freetype2
    pkgs.fontconfig
    pkgs.pkgconfig
  ];

  dontConfigure = true;
  installPhase = ''
    mkdir -p $out/include/
    cp -r ./*.h $out/include/
    mkdir -p $out/lib/
    cp -r ./lib/* $out/lib/
  '';
  checkPhase = "";
  passthru = { backendPath = name; };
}
