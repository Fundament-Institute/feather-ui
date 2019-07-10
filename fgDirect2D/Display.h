// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef D2D__DISPLAY_H
#define D2D__DISPLAY_H

#include "feather/backend.h"

namespace D2D {
  struct Direct2D;
  struct Context;

  struct DisplayData
  {
    fgVec dpi;
    float scale;
  };

  fgMessageResult DisplayBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);
}

#endif