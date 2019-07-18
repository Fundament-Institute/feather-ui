// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__BOX_H
#define FG__BOX_H

#include "../message.h"
#include "../backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Contains additional outline information for a "box" behavior function
typedef struct
{
  const fgRect corners;
  fgColor fillColor;
  float border;
  fgColor borderColor;
  float blur;
  fgAsset* asset;
} fgBoxData;

FG_COMPILER_DLLEXPORT fgMessageResult fgBoxBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);
FG_COMPILER_DLLEXPORT unsigned int fgBoxResolver(void* ptr, unsigned int index, fgCalcNode* out, const char* id);

#ifdef  __cplusplus
}
#endif

#endif