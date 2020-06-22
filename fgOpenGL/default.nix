{ config ? { }, lib ? { }, pkgs ? import <nixpkgs> { }
, feather ? pkgs.callPackage ../. { }, ... }:

let inherit (pkgs) stdenv;
in stdenv.mkDerivation {
  name = "fgOpenGL";
  version = "0.0.1";

  src = ./.;
  buildInputs = [ feather.backendInterface ];

  dontConfigure = true;
  installPhase = ''
    mkdir -p $out #replaceme with an install command
  '';
  checkPhase = "";

}
