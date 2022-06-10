// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__FORMAT_H
#define GL__FORMAT_H

#include "compiler.hpp"
#include "backend.h"
#include "glad/glad.h"

namespace GL {
  // Struct representing an OpenGL format triplet because the OpenGL spec is awful
  struct Format
  {
    GLint internalformat;
    GLenum components;
    GLenum type;

    static Format Map(GLint internalformat) noexcept;
    static Format Create(unsigned char format, bool sRGB) noexcept;
  };
}

#endif