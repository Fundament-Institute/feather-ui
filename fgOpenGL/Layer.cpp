// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include <assert.h>
#include <math.h>

using namespace GL;

Layer::Layer(FG_Vec s, int f, Context* c) : RenderTarget(s, f, FG_PixelFormat_R8G8B8A8_UNORM, c), opacity(0)
{
  format = FG_Format_LAYER;
  blend  = Context::PREMULTIPLY_BLEND;
  mat4x4_identity(transform);
}

Layer::~Layer() { }

void Layer::Update(float* tf, float o, FG_BlendState* b)
{
  opacity = o;
  if(tf)
    MEMCPY(transform, sizeof(transform), tf, 16 * sizeof(float));
  if(b)
    blend = *b;
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