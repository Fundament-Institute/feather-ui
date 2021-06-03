// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__RT_H
#define GL__RT_H

#include "compiler.h"
#include "backend.h"
#include "linmath.h"

namespace GL {
  struct Context;

  struct RenderTarget : FG_Asset
  {
    RenderTarget(FG_Vec s, int f, uint8_t format, Context* c);
    ~RenderTarget();
    // Moves the texture to another window if necessary
    bool Set(Context* context);
    void Destroy();
    bool Create();

    static void mat4x4_proj(mat4x4 M, float l, float r, float b, float t, float n, float f);
    static const float NEARZ;
    static const float FARZ;

    unsigned int framebuffer;
    uint8_t pixelformat;
    mat4x4 proj;
    Context* context;
    bool initialized;
  };
}

#endif
