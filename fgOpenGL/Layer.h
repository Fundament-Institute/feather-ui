// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__LAYER_H
#define GL__LAYER_H

#include "backend.h"
#include "linmath.h"

namespace GL {
  struct Context;

  struct Layer : FG_Asset
  {
    Layer(FG_Vec s, int f, Context* c);
    ~Layer();
    // Moves the texture to another window if necessary
    bool Update(float* tf, float o, FG_BlendState* blend, Context* context);
    void Destroy();
    bool Create();
    int Composite();

    static void mat4x4_proj(mat4x4 M, float l, float r, float b, float t, float n, float f);
    static const float NEARZ;
    static const float FARZ;

    unsigned int framebuffer;
    mat4x4 transform;
    mat4x4 proj;
    float opacity;
    Context* context;
    bool initialized;
    FG_BlendState blend;
  };
}

#endif
