// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__ASSET_H
#define GL__ASSET_H

#include "backend.h"

namespace GL {
  // Internal format trackers
  enum Format : int
  {
    FORMAT_UNKNOWN  = FG_Format_UNKNOWN,
    FORMAT_VERTEX   = (1 << 8) | FG_Format_UNKNOWN,
    FORMAT_GRADIENT = (2 << 8) | FG_Format_UNKNOWN,
  };

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