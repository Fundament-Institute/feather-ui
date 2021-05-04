# Feather

## Building without Nix

### Graphics Backend

Build the graphics backend:

```
mkdir build
cd build
cmake ..
make
```

### cpptest

First, build the backend. Then, from the root of the project:

```
# enter cpptest directory
cd cpptest

# build it
CPPFLAGS='-I../include' LDFLAGS='-L../bin-x64' BINDIR=bin OBJDIR=bin/obj LIBDIR=lib make

# run it
LD_LIBRARY_PATH='../bin-x64' bin/cpptest
```

### everything else

Unknown

## Running OpenGL Executables in Nix

### Linux (non NixOS)

1. Install the appropriate version of [nixGL](https://github.com/guibou/nixGL) for your graphics card driver.

2. Use `nixGLXXX` to run executables as in the example below.

```
# Using proprietary Nvidia backend, adjust for your graphics driver.
nixGLNvidia ./result/bin/fgTests
```