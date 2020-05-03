{ pkgs ? import <nixpkgs> { } }:

with pkgs;

let
  llvmPackages = pkgs.llvmPackages_9;
  luajitCommit = "9143e86498436892cb4316550be4d45b68a61224";
  luajitArchive = "LuaJIT-${luajitCommit}.tar.gz";
  luajitSrc = fetchurl {
    url = "https://github.com/LuaJIT/LuaJIT/archive/${luajitCommit}.tar.gz";
    sha256 = "0kasmyk40ic4b9dwd4wixm0qk10l88ardrfimwmq36yc5dhnizmy";
  };
  terra = pkgs.terra.overrideAttrs (old: {
    src = pkgs.fetchFromGitHub {
      owner = "terralang";
      repo = "terra";
      rev = "451ec80cfdaf34dbf6cb7fba8cba4ea7353b4cff";
      sha256 = "0i91058r9vrwdvcsf1xm936a6f19vy200aynlbwb1kwdnlnci1m2";
    };

    buildInputs = with llvmPackages; [ llvm clang-unwrapped pkgs.ncurses ];

    preBuild = ''
      cat >Makefile.inc<<EOF
      CLANG = ${stdenv.lib.getBin llvmPackages.clang-unwrapped}/bin/clang
      LLVM_CONFIG = ${stdenv.lib.getBin llvmPackages.llvm}/bin/llvm-config
      EOF

      mkdir -p build
      cp ${luajitSrc} build/${luajitArchive}
      echo build/${luajitArchive}
    '';
  });
  terraIncludes = lst:
    builtins.concatStringsSep ";" (map (pkg: "${pkg}/include") lst);
  terraLibs = lst: builtins.concatStringsSep ";" (map (pkg: "${pkg}/lib") lst);
  terraShell = { buildInputs, ... }@args:
    mkShell (args // {
      INCLUDE_PATH = terraIncludes buildInputs;
      LIBRARY_PATH = terraLibs buildInputs;
      buildInputs = [ terra ] ++ buildInputs;
    });
in terraShell { buildInputs = [ pkgs.libiconv pkgs.glibc.dev ]; }
