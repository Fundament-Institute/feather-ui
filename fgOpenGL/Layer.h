// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__LAYER_H
#define GL__LAYER_H

#include "backend.h"

struct GLFWwindow;

namespace GL {
  struct Layer
  {
    Layer(FG_Rect a, float* tf, float o, GLFWwindow* window);
    ~Layer();
    // Reallocates the texture if necessary
    bool Update(FG_Rect a, float* tf, float o, GLFWwindow* window);
    void Dirty(const FG_Rect* area);
    void Destroy();
    bool Create();

    unsigned int framebuffer;
    unsigned int texture;
    FG_Rect area;
    float transform[16];
    float opacity;
    GLFWwindow* window;
    FG_Rect dirty; // Stores how much of the layer is actually dirty. If nothing is dirty, we ignore all draw calls.
    bool initialized;
  };
}

#endif
