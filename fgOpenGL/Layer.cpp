// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "Backend.h"
#include <assert.h>

using namespace GL;

Layer::Layer(FG_Rect a, float* tf, float o, GLFWwindow* w) : area(a), dirty(a), opacity(o), window(w)
{
  MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));
  Create();
}

Layer::~Layer() { Destroy(); }

void Layer::Destroy()
{
  glDeleteFramebuffers(1, &framebuffer);
  glDeleteTextures(1, &texture);
}

bool Layer::Create()
{
  glGenFramebuffers(1, &framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, area.right - area.left, area.bottom - area.top, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    return false;

  return true;
}

bool Layer::Update(FG_Rect a, float* tf, float o, GLFWwindow* w)
{
  opacity = o;
  if(w != window) // If true we have to go BACK to the original window context, delete everything, then recreate it
  {
    auto cur = glfwGetCurrentContext();
    glfwMakeContextCurrent(window);
    Destroy();
    glfwMakeContextCurrent(cur);

    window = w;
    dirty  = a;
    area   = a;
    return Create();
  }

  window = w;
  if(tf)
    MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));

  // Resize the texture
  if(a.left != area.left || a.top != area.top || a.right != area.right || a.bottom != area.bottom)
  {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, area.right - area.left, area.bottom - area.top, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    area  = a;
    dirty = a;
  }

  return true;
}