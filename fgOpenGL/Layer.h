// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__LAYER_H
#define GL__LAYER_H

#include "backend.h"

struct GLFWwindow;

namespace GL {
  struct Layer : FG_Asset
  {
    Layer(FG_Vec s, GLFWwindow* w);
    ~Layer();
    // Moves the texture to another window if necessary
    bool Update(float* tf, float o, GLFWwindow* window);
    void Destroy();
    bool Create();

    unsigned int framebuffer;
    float transform[16];
    float opacity;
    GLFWwindow* window;
    bool initialized;
  };
}

#endif
