{ pkgs ? import <nixpkgs> { } }:

with pkgs;
with callPackage ./lib.nix { };
terraShell {
  buildInputs = [ pkgs.libiconv pkgs.glibc.dev ];
  TERRA_PATH = "./?.t;./?/init.t;deps/?.t;deps/?.init.t";
  LUA_PATH = "./?.lua;deps/?.lua";
  FEATHER_BACKEND = let
    backend =
      (callPackage ./fgOpenGL { inherit SOIL harfbuzz glfw freetype2; });
  in "${backend}/lib/lib${backend.backendPath}.so";
}
