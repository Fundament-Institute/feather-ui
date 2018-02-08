// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RESOURCE_H__
#define __FG_RESOURCE_H__

#include "fgElement.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGRESOURCE_FLAGS
{
  FGRESOURCE_UVTILE = (1 << 9),
  FGRESOURCE_RECT = (1 << 10), // Indicates this is a rounded rectangle
  FGRESOURCE_CIRCLE = (2 << 10), // Indicates this is a circle
  FGRESOURCE_TRIANGLE = (3 << 10), // Indicates this is a triangle
  FGRESOURCE_SHAPEMASK = (3 << 10),
};

// fgResource stores a renderable image/vector/shader, that optionally has UV coordinates.
typedef struct {
  fgElement element;
  CRect uv;
  fgAsset asset;
  FABS outline; // for rounded rectangles and circles, this specifies the width of the outline
  fgColor color;
  fgColor edge;
#ifdef  __cplusplus
  inline operator fgElement*() { return &element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgResource;

FG_EXTERN fgElement* fgResource_Create(fgAsset asset, const CRect* uv, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN void fgResource_Init(fgResource* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN void fgResource_Destroy(fgResource* self);
FG_EXTERN size_t fgResource_Message(fgResource* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif