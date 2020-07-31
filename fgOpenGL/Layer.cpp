// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "Backend.h"
#include <assert.h>
#include <math.h>

using namespace GL;

Layer::Layer(FG_Rect a, float* tf, float o, GLFWwindow* w) : area(a), dirty(a), opacity(o), window(w), initialized(false)
{
  MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));
  initialized = Create();
}

Layer::~Layer() { Destroy(); }

void Layer::Destroy()
{
  if(initialized)
  {
    glDeleteFramebuffers(1, &framebuffer);
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

  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)ceilf(area.right - area.left), (GLsizei)ceilf(area.bottom - area.top), 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
    initialized = Create();
    return initialized;
  }

  window = w;
  if(tf)
    MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));

  // Resize the texture
  if(a.left != area.left || a.top != area.top || a.right != area.right || a.bottom != area.bottom)
  {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)ceilf(area.right - area.left), (GLsizei)ceilf(area.bottom - area.top),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    area  = a;
    dirty = a;
  }

  return true;
}

void Layer::Dirty(const FG_Rect* rect)
{
  if(!rect)
    dirty = area;
  else if(dirty.left != NAN || dirty.top != NAN || dirty.right != NAN || dirty.bottom != NAN)
    dirty = *rect;
  else
  {
    if(rect->left < dirty.left)
      dirty.left = rect->left;
    if(rect->top < dirty.top)
      dirty.top = rect->top;
    if(rect->right > dirty.right)
      dirty.right = rect->right;
    if(rect->bottom > dirty.bottom)
      dirty.bottom = rect->bottom;
  }
}
