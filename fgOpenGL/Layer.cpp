// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include <assert.h>
#include <math.h>

using namespace GL;

const float Layer::NEARZ = 0.2f;
const float Layer::FARZ  = 100.0f;

Layer::Layer(FG_Vec s, int f, Context* c) : context(c), opacity(0), initialized(false)
{
  flags  = f;
  format = FG_Format_LAYER;
  blend  = Context::PREMULTIPLY_BLEND;
  mat4x4_identity(transform);
  size.x = static_cast<int>(ceilf(s.x));
  size.y = static_cast<int>(ceilf(s.y));
  Layer::mat4x4_proj(proj, 0, static_cast<float>(size.x), static_cast<float>(size.y), 0, NEARZ, FARZ);

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

// Creates the layer on the current openGL context based on the assetflags assigned to this layer object
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

  if(flags & FG_AssetFlags_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  backend->LogError("glTexImage2D");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  backend->LogError("glTexParameteri");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  backend->LogError("glTexParameteri");

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
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

// This assembles a custom projection matrix specifically designed for 2D drawing.
void Layer::mat4x4_proj(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
  memset(M, 0, sizeof(mat4x4));

  M[0][0] = 2.0f / (r - l);
  M[0][1] = 0.f;
  M[0][2] = 0.f;
  M[0][3] = 0.f;

  M[1][0] = 0.f;
  M[1][1] = 2.0f / (t - b);
  M[1][2] = 0.f;
  M[1][3] = 0.f;

  M[2][0] = 0.f;
  M[2][1] = 0.f;
  M[2][2] = -((f + n) / (f - n));
  M[2][3] = -1.0f;

  M[3][0] = -(r + l) / (r - l);
  M[3][1] = -(t + b) / (t - b);
  M[3][2] = -((2.f * f * n) / (f - n));
  M[3][3] = 0.f;

  mat4x4_translate_in_place(M, 0, 0, -1.0f);
}

// Assembles a Quad mesh and calls DrawTextureQuad with this layer's backing texture, while also applying
// this layer's opacity to the alpha channel color modulation.
int Layer::Composite()
{
  // Our quad mesh is the real pixel size of the layer, but has [0,1] UV coordinates.
  ImageVertex v[4];

  v[0].posUV[0] = 0;
  v[0].posUV[1] = 0;
  v[0].posUV[2] = 0;
  v[0].posUV[3] = 1.0f;

  v[1].posUV[0] = static_cast<float>(size.x);
  v[1].posUV[1] = 0;
  v[1].posUV[2] = 1.0f;
  v[1].posUV[3] = 1.0f;

  v[2].posUV[0] = 0;
  v[2].posUV[1] = static_cast<float>(size.y);
  v[2].posUV[2] = 0;
  v[2].posUV[3] = 0;

  v[3].posUV[0] = static_cast<float>(size.x);
  v[3].posUV[1] = static_cast<float>(size.y);
  v[3].posUV[2] = 1.0f;
  v[3].posUV[3] = 0;

  mat4x4 mvp;
  mat4x4_mul(mvp, context->GetProjection(), transform);

  context->ApplyBlend(&blend);
  return context->DrawTextureQuad(data.index, v, FG_Color{ 0x00FFFFFF + ((unsigned int)roundf(0xFF * opacity) << 24) }, mvp,
                                  false);
}