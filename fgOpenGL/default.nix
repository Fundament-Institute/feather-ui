{ config ? { }, lib ? { }, pkgs ? import <nixpkgs> { }
, feather ? pkgs.callPackage ../. { }, ... }:

let inherit (pkgs) stdenv;
in stdenv.mkDerivation {
  name = "fgOpenGL";
  version = "0.0.1";
  makeflags = [ "BINDIR=bin" "LIBDIR=lib" "OBJDIR=bin/obj" ];
  src = ./.;
  CXXLD="$(CXX)";
  CPPFLAGS = "-Wall -Wshadow -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch";

  buildInputs = [ feather.backendInterface feather.SOIL feather.harfbuzz feather.glfw feather.freetype2 ];
  
  dontConfigure = true;
  installPhase = ''
    mkdir -p $out/include/
    cp -r ./*.h $out/include/
    mkdir -p $out/lib/
    cp -r ./lib/* $out/lib/
  '';
  checkPhase = "";
}
