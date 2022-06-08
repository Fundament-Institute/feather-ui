// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#ifndef GL__ENUM_MAPPING_H
#define GL__ENUM_MAPPING_H

#include "backend.h"
#include "glad/glad.h"

namespace GL {
  static constexpr uint16_t ComparisonMapping[] = { 0,          GL_NEVER,    GL_LESS,   GL_EQUAL, GL_LEQUAL,
                                                    GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS };
  static constexpr uint16_t PrimitiveMapping[]  = { GL_POINTS,
                                                   GL_LINES,
                                                   GL_TRIANGLES,
                                                   GL_LINE_STRIP,
                                                   GL_TRIANGLE_STRIP,
                                                   GL_LINES_ADJACENCY,
                                                   GL_TRIANGLES_ADJACENCY,
                                                   GL_LINE_STRIP_ADJACENCY,
                                                   GL_TRIANGLE_STRIP_ADJACENCY };
  static constexpr uint16_t BlendOpMapping[] = { GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX };
  static constexpr uint16_t BlendMapping[]   = {
    GL_ZERO,
    GL_ONE,
    GL_SRC_COLOR,
    GL_ONE_MINUS_SRC_COLOR,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_DST_ALPHA,
    GL_ONE_MINUS_DST_ALPHA,
    GL_DST_COLOR,
    GL_ONE_MINUS_DST_COLOR,
    GL_SRC_ALPHA_SATURATE,
    GL_CONSTANT_COLOR,
    GL_ONE_MINUS_CONSTANT_COLOR,
    GL_SRC1_COLOR,
    GL_ONE_MINUS_SRC1_COLOR,
    GL_SRC1_ALPHA,
    GL_ONE_MINUS_SRC1_ALPHA,
  };

  static constexpr uint16_t ShaderTypeMapping[] = { GL_HALF_FLOAT,   GL_FLOAT,        GL_DOUBLE, GL_INT,
                                                    GL_UNSIGNED_INT, GL_UNSIGNED_INT, 0,         0 };

  static constexpr uint16_t TypeMapping[] = {
    0,
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_PIXEL_PACK_BUFFER,
    GL_PIXEL_UNPACK_BUFFER,
    GL_COPY_READ_BUFFER,
    GL_COPY_WRITE_BUFFER,
    GL_TEXTURE_BUFFER,
    GL_TRANSFORM_FEEDBACK_BUFFER,
    GL_UNIFORM_BUFFER,
    GL_TEXTURE_1D,
    GL_TEXTURE_2D,
    GL_TEXTURE_3D,
    GL_TEXTURE_2D_MULTISAMPLE,
    GL_PROXY_TEXTURE_2D_MULTISAMPLE,
  };

  static constexpr uint16_t StencilOpMapping[] = {
    0, GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP,
  };

  static constexpr uint16_t ShaderStageMapping[] = {
    GL_FRAGMENT_SHADER,
    GL_VERTEX_SHADER,
    GL_GEOMETRY_SHADER,
    #if 0 // We don't support these shaders yet
                                                    GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_COMPUTE_SHADER,
    #endif
  };

  template<class T, int SIZE> constexpr int ArraySize(T (&)[SIZE]) { return SIZE; }
}

#endif