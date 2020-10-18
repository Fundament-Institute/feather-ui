// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include <assert.h>
#include <math.h>

using namespace GL;

Layer::Layer(FG_Vec s, GLFWwindow* w) : window(w), initialized(false)
{
  format      = FG_Format_LAYER;
  size.x      = s.x;
  size.y      = s.y;
  glfwGetWindowContentScale(w, &dpi.x, &dpi.y);
  initialized = Create();
}

Layer::~Layer() { Destroy(); }

void Layer::Destroy()
{
  if(initialized)
  {
    glDeleteFramebuffers(1, &framebuffer);
    GLuint texture = data.index;
    glDeleteTextures(1, &texture);
  }
  initialized = false;
}

bool Layer::Create()
{
  if(initialized)
    return true;

  glGenFramebuffers(1, &framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)ceilf(size.x), (GLsizei)ceilf(size.y), 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    return false;

  data.index = texture;
  return true;
}

bool Layer::Update(float* tf, float o, GLFWwindow* w)
{
  opacity = o;
  if(w != window) // If true we have to go BACK to the original window context, delete everything, then recreate it
  {
    auto cur = glfwGetCurrentContext();
    glfwMakeContextCurrent(window);
    Destroy();
    glfwMakeContextCurrent(cur);

    window = w;
    initialized = Create();
    return initialized;
  }

  window = w;
  if(tf)
    MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));

  return true;
}