# Feather

Feather is a language-agnostic, universal user interface library.

## Building

Feather uses `cmake` to build, and utilizes nix flakes on NixOS, or the `vcpkg` build system on all other operating systems.

### vcpkg

Feather maintains a fork of `vcpkg` (this may be turned into an overlay in the future). Clone the git repository from [https://github.com/Fundament-Software/vcpkg](https://github.com/Fundament-Software/vcpkg) into a folder with enough space to hold a bunch of compiled binaries, and run `bootstrap-vcpkg.bat` (for windows) or `bootstrap-vcpkg.sh` (for other OSes). Once the script completes, run `vcpkg install glfw3:x64-windows-static` (on windows, start with `.\vcpkg.exe`) and follow any instructions that pop up.

Next, clone the featherGUI repo, if you haven't already. Inside the root directory of the repo, create a `build` directory and open it. Inside the empty `build` directory, run the `cmake` command that is appropriate for your platform, being sure to replace `<ROOT_VCPKG_FOLDER>` with the folder you cloned `vcpkg` into, and adding any additional options you would like.

#### Windows

    cmake .. -DCMAKE_TOOLCHAIN_FILE=<ROOT_VCPKG_FOLDER>\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static

If you don't have cmake installed on your system, you can use the version `vcpkg` installed:

    <ROOT_VCPKG_FOLDER>\downloads\tools\cmake-<LATEST_CMAKE_VERSION>-windows\cmake-<LATEST_CMAKE_VERSION>-windows-x86\bin\cmake.exe .. -DCMAKE_TOOLCHAIN_FILE=<ROOT_VCPKG_FOLDER>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static

#### Other Platforms

    cmake .. -DCMAKE_TOOLCHAIN_FILE=<ROOT_VCPKG_FOLDER>/scripts/buildsystems/vcpkg.cmake
    
If you don't have cmake installed on your system, you can use the version `vcpkg` installed:

    <ROOT_VCPKG_FOLDER>/downloads/tools/<CMAKE_PATH_FOR_YOUR_PLATFORM>/bin/cmake .. -DCMAKE_TOOLCHAIN_FILE=<ROOT_VCPKG_FOLDER>/scripts/buildsystems/vcpkg.cmake

### Nix Flake

Ensure that you have enabled flake support in your `configuration.nix` file (or equivilent):

    nix.package = pkgs.nixFlakes;
    nix.extraOptions = ''
      experimental-features = nix-command flakes
    '';

Clone the featherGUI repository, if you haven't already, then run `nix build` inside the root directory.