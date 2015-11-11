// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_H__
#define __FG_LAYOUT_H__

#include "fgSkin.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG_CLASS_LAYOUT {
  char* name;
  fgElement element;
  fgFlag flags;
  fgStyle style; // style overrides
  fgVector children; // Type: fgClassLayout
  fgVector statics; // Type: fgFullDef
} fgClassLayout;

typedef struct FG_LAYOUT {
  fgVector resources; // Type: void*
  fgVector layout; // Type: fgClassLayout
} fgLayout;

FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadFileUBJSON(const char* file);
FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadUBJSON(const void* data, FG_UINT length);
FG_EXTERN void FG_FASTCALL fgLayout_Destroy(fgLayout* layout);
FG_EXTERN void FG_FASTCALL fgLayout_Free(fgLayout* layout);

#ifdef  __cplusplus
}
#endif

#endif