// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include <assert.h>
#include <math.h>

using namespace GL;

Layer::Layer(FG_Vec s, Context* c) : context(c), opacity(0), initialized(false)
{
  format      = FG_Format_LAYER;
  size.x      = s.x;
  size.y      = s.y;
  if(c->GetWindow())
    glfwGetWindowContentScale(c->GetWindow(), &dpi.x, &dpi.y);
  initialized = Create();
}

Layer::~Layer() { Destroy(); }

void Layer::Destroy()
{
  if(initialized)
  {
    glDeleteFramebuffers(1, &framebuffer);
    context->GetBackend()->LogError("glDeleteFramebuffers");
    GLuint texture = data.index;
    glDeleteTextures(1, &texture);
    context->GetBackend()->LogError("glDeleteTextures");
  }
  initialized = false;
}

bool Layer::Create()
{
  if(initialized)
    return true;
  
  auto backend = context->GetBackend();

  glGenFramebuffers(1, &framebuffer);
  backend->LogError("glGenFramebuffers");
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  backend->LogError("glBindFramebuffer");

  GLuint texture;
  glGenTextures(1, &texture);
  backend->LogError("glGenTextures");
  glBindTexture(GL_TEXTURE_2D, texture);
  backend->LogError("glBindTexture");
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)ceilf(size.x), (GLsizei)ceilf(size.y), 0,
               GL_RGBA, GL_UNSIGNED_BYTE, 0);
  backend->LogError("glTexImage2D");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  backend->LogError("glTexParameteri");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  backend->LogError("glTexParameteri");

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0);
  backend->LogError("glFramebufferTexture");

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  data.index = texture;
  return true;
}

bool Layer::Update(float* tf, float o, FG_BlendState* b, Context* c)
{
  opacity = o;
  if(tf)
    MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));
  if(b)
    blend = *b;

  if(c != context) // If true we have to go BACK to the original window context, delete everything, then recreate it
  {
    auto cur = glfwGetCurrentContext();
    if(context->GetWindow())
      glfwMakeContextCurrent(context->GetWindow());
    Destroy();
    glfwMakeContextCurrent(cur);

    context     = c;
    initialized = Create();
    return initialized;
  }

  context = c;
  return true;
}