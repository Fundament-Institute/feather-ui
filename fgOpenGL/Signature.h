// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__SHADER_INPUT_H
#define GL__SHADER_INPUT_H

#include "VAO.h"

namespace GL {
  struct Context;
  struct Shader;

  // input signature for a shader, represented as a VAO
  struct Signature
  {
    Signature(Backend* backend, std::span<FG_Asset*> buffers, std::span<FG_ShaderParameter> parameters);
    ~Signature();

    std::vector<FG_Asset*> _buffers;
    std::vector<GLuint> _strides;
    std::vector<FG_ShaderParameter> _parameters;
  };
}

#endif
