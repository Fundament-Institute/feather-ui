// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__ASSET_H
#define GL__ASSET_H

#include "compiler.h"
#include "backend.h"

namespace GL {
  struct Asset : FG_Asset
  {
    int channels;
  };

  struct QuadVertex
  {
    float pos[2];
  };

  struct ImageVertex
  {
    float posUV[4];
    float color[4];
  };
}

#endif