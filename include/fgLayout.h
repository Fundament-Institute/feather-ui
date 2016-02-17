// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_H__
#define __FG_LAYOUT_H__

#include "fgSkin.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG_CLASS_LAYOUT {
  fgStyleLayout style;
  fgVector children; // Type: fgClassLayout
} fgClassLayout;

typedef struct FG_LAYOUT {
  fgVector resources; // Type: void*
  fgVector fonts; // Type: void*
  fgVector layout; // Type: fgClassLayout
} fgLayout;

FG_EXTERN void FG_FASTCALL fgLayout_Init(fgLayout* self);
FG_EXTERN void FG_FASTCALL fgLayout_Destroy(fgLayout* self);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource);
FG_EXTERN void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font);
FG_EXTERN void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font);
FG_EXTERN FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* name, fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout);
FG_EXTERN fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout);

FG_EXTERN void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* name, fgElement* element, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self);
FG_EXTERN FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* name, fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child);
FG_EXTERN fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child);

FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadFileUBJSON(const char* file);
FG_EXTERN fgLayout* FG_FASTCALL fgLayout_LoadUBJSON(const void* data, FG_UINT length);
FG_EXTERN fgChild* fgLayoutLoadMapping(const char* name, fgFlag flags, fgChild* parent, fgChild* prev, fgElement* element);

#ifdef  __cplusplus
}
#endif

#endif