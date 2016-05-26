// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_LAYOUT_H__
#define _FG_LAYOUT_H__

#include "fgSkin.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_CLASS_LAYOUT;
typedef fgDeclareVector(struct _FG_CLASS_LAYOUT, ClassLayout) fgVectorClassLayout;

typedef struct _FG_CLASS_LAYOUT {
  fgStyleLayout style;
  fgVectorClassLayout children; // Type: fgClassLayout
} fgClassLayout;

typedef struct _FG_LAYOUT {
  fgVector resources; // Type: void*
  fgVector fonts; // Type: void*
  fgVectorClassLayout layout; // Type: fgClassLayout
} fgLayout;

FG_EXTERN void FG_FASTCALL fgLayout_Init(fgLayout* self);
FG_EXTERN void FG_FASTCALL fgLayout_Destroy(fgLayout* self);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource);
FG_EXTERN void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font);
FG_EXTERN void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order = 0);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout);
FG_EXTERN fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout);

FG_EXTERN void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order = 0);
FG_EXTERN void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self);
FG_EXTERN FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order = 0);
FG_EXTERN char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child);
FG_EXTERN fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child);

FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadFileUBJSON(const char* file);
FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadUBJSON(const void* data, FG_UINT length);

FG_EXTERN size_t FG_FASTCALL fgLayout_Default(fgElement* self, const FG_Msg* msg, CRect* area, AbsRect* parent);
FG_EXTERN size_t FG_FASTCALL fgLayout_Distribute(fgElement* self, const FG_Msg* msg, char axis);
FG_EXTERN size_t FG_FASTCALL fgLayout_Tile(fgElement* self, const FG_Msg* msg, char axes, CRect* area);

#ifdef  __cplusplus
}
#endif

#endif