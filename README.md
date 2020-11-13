# Feather

## Running OpenGL Executables

### Linux (non NixOS)

1. Install the appropriate version of [nixGL](https://github.com/guibou/nixGL) for your graphics card driver.

2. Use `nixGLXXX` to run executables as in the example below.

```
# Using proprietary Nvidia backend, adjust for your graphics driver.
nixGLNvidia ./result/bin/fgTests
```