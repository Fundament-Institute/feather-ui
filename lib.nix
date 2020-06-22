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
      rev = "acce1f174bc06a1252fa4fd065e7681e11e324f8";
      sha256 = "1n82n1gdqw2g33j7cf09gr4r6jqp7xg820psj7j1x4xc6l2bfryy";
      # date = 2020-05-26T21:44:09-07:00;
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
in { inherit terraShell terra; }
