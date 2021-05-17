{ config ? { }, lib ? { }, pkgs ? import <nixpkgs> { }
, feather ? pkgs.callPackage ../. { }, ... }:

let inherit (pkgs) stdenv;
in stdenv.mkDerivation rec {
  name = "cpptest";
  version = "0.0.1";
  makeFlags = [ "BINDIR=bin" "LIBDIR=lib" "OBJDIR=bin/obj" ];
  src = ./.;
  CPPFLAGS =
    "-I. -Wall -Wshadow -Wno-attributes -Wno-unknown-pragmas -Wno-missing-braces -Wno-unused-function -Wno-comment -Wno-char-subscripts -Wno-sign-compare -Wno-unused-variable -Wno-switch -std=c++17 -msse -msse2 -msse3 -mmmx -m3dnow -mcx16";

  buildInputs = [
    feather.backendInterface
    feather.backends
    pkgs.pkgconfig
    pkgs.libglvnd
  ];

  dontStrip = true;
  dontConfigure = true;
  installPhase = ''
    mkdir -p $out/bin/
    cp -r ./bin/* $out/bin/
  '';
  checkPhase = "";
  passthru = { backendPath = name; };
}
