// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RESOURCE_H__
#define __FG_RESOURCE_H__

#include "fgChild.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGRESOURCE_FLAGS
{
  FGRESOURCE_STRETCHX = (1 << 6), // Stretches the image instead of tiling it
  FGRESOURCE_STRETCHY = (1 << 7),
};

// fgResource stores a renderable image/vector/shader, that optionally has UV coordinates.
typedef struct {
  fgChild element;
  CRect uv;
  void* res;
  unsigned int color;
} fgResource;

FG_EXTERN fgChild* FG_FASTCALL fgResource_Create(void* res, const CRect* uv, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgResource_Init(fgResource* self, void* res, const CRect* uv, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgResource_Destroy(fgResource* self);
FG_EXTERN size_t FG_FASTCALL fgResource_Message(fgResource* self, const FG_Msg* msg);
void FG_FASTCALL fgResource_Recalc(fgResource* self);

FG_EXTERN void* FG_FASTCALL fgCreateResourceFile(fgFlag flags, const char* file);
FG_EXTERN void* FG_FASTCALL fgCreateResource(fgFlag flags, const char* data, size_t length);
FG_EXTERN void* FG_FASTCALL fgCloneResource(void* res);
FG_EXTERN void FG_FASTCALL fgDestroyResource(void* res);
FG_EXTERN void FG_FASTCALL fgDrawResource(void* res, const CRect* uv, unsigned int color, const AbsRect* area, FABS rotation, AbsVec* center, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgResourceSize(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgDrawLine(AbsVec p1, AbsVec p2, unsigned int color);

#ifdef  __cplusplus
}
#endif

#endif