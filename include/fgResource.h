// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_RESOURCE_H__
#define _FG_RESOURCE_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGRESOURCE_FLAGS
{
  FGRESOURCE_UVTILE = (1 << 8),
  FGRESOURCE_ROUNDRECT = (1 << 9), // Indicates this is a rounded rectangle
  FGRESOURCE_CIRCLE = (2 << 9), // Indicates this is a circle
  FGRESOURCE_SHAPEMASK = (3 << 9),
};

// fgResource stores a renderable image/vector/shader, that optionally has UV coordinates.
typedef struct {
  fgElement element;
  CRect uv;
  void* res;
  FABS outline; // for rounded rectangles and circles, this specifies the width of the outline
  fgColor color;
  fgColor edge;
#ifdef  __cplusplus
  inline operator fgElement*() { return &element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgResource;

FG_EXTERN fgElement* FG_FASTCALL fgResource_Create(void* res, const CRect* uv, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgResource_Init(fgResource* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgResource_Destroy(fgResource* self);
FG_EXTERN size_t FG_FASTCALL fgResource_Message(fgResource* self, const FG_Msg* msg);
void FG_FASTCALL fgResource_Recalc(fgResource* self);

FG_EXTERN void* FG_FASTCALL fgCreateResourceFile(fgFlag flags, const char* file);
FG_EXTERN void* FG_FASTCALL fgCreateResource(fgFlag flags, const char* data, size_t length);
FG_EXTERN void* FG_FASTCALL fgCloneResource(void* res);
FG_EXTERN void FG_FASTCALL fgDestroyResource(void* res);
FG_EXTERN void FG_FASTCALL fgDrawResource(void* res, const CRect* uv, unsigned int color, unsigned int edge, FABS outline, const AbsRect* area, FABS rotation, const AbsVec* center, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgResourceSize(void* res, const CRect* uv, AbsVec* dim, fgFlag flags);

#ifdef  __cplusplus
}
#endif

#endif