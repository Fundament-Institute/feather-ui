// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__FORMAT_H
#define GL__FORMAT_H

#include "compiler.h"
#include "backend.h"
#include "glad/gl.h"

namespace GL {
  // Struct representing an OpenGL format triplet because the OpenGL spec is awful
  struct GLFormat
  {
    GLint internalformat;
    GLenum components;
    GLenum type;

    static GLFormat Map(GLint internalformat);
    static GLFormat Create(uint8_t format, bool sRGB);
  };
}

#endif