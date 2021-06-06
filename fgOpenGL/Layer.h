// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__LAYER_H
#define GL__LAYER_H

#include "RenderTarget.h"

namespace GL {
  struct Context;

  struct Layer : RenderTarget
  {
    Layer(FG_Vec s, int f, Context* c);
    ~Layer();
    // Moves the texture to another window if necessary
    void Update(float* tf, float o, FG_BlendState* blend);
    int Composite();

    mat4x4 transform;
    float opacity;
    FG_BlendState blend;
  };
}

#endif
