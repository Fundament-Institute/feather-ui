{ pkgs ? import <nixpkgs> { } }:

with pkgs;
with callPackage ./lib.nix { };
terraShell { buildInputs = [ pkgs.libiconv pkgs.glibc.dev ]; }
