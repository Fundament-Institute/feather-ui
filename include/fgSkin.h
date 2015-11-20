// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SKIN_H__
#define __FG_SKIN_H__

#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct __FG_STYLE_MSG
{
  FG_Msg msg;
  struct __FG_STYLE_MSG* next;
} fgStyleMsg;

typedef struct __FG_STYLE
{
  ptrdiff_t index; // Index of the child this applies to (only used for substyles). Negative numbers map to existing statics, equal to index + prechild.
  fgStyleMsg* styles;
  fgVector substyles; // type: fgStyle
} fgStyle;

typedef struct FG_STYLE_LAYOUT {
  char* name;
  fgElement element;
  fgFlag flags;
  fgStyle style; // style overrides
} fgStyleLayout;

typedef struct __FG_SKIN
{
  int index; // index of the subskin, also equal to index + prechild. Should never be positive.
  fgVector resources; // type: void*
  fgVector fonts;
  fgVector children; // type: fgStyleLayout
  fgVector styles; // type: fgStyle
  fgVector subskins; // type: fgSkin (should be used for prechildren ONLY)
  struct __kh_fgSkins_t* skinmap;
} fgSkin;

struct __kh_fgSkins_t;

FG_EXTERN void FG_FASTCALL fgSkin_Init(fgSkin* self);
FG_EXTERN void FG_FASTCALL fgSkin_Destroy(fgSkin* self);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddResource(fgSkin* self, void* resource);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveResource(fgSkin* self, FG_UINT resource);
FG_EXTERN void* FG_FASTCALL fgSkin_GetResource(fgSkin* self, FG_UINT resource);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddFont(fgSkin* self, void* font);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveFont(fgSkin* self, FG_UINT font);
FG_EXTERN void* FG_FASTCALL fgSkin_GetFont(fgSkin* self, FG_UINT font);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddChild(fgSkin* self, char* name, fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child);
FG_EXTERN fgStyleLayout* FG_FASTCALL fgSkin_GetChild(fgSkin* self, FG_UINT child);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style);
FG_EXTERN fgStyle* FG_FASTCALL fgSkin_GetStyle(fgSkin* self, FG_UINT style);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddSubSkin(fgSkin* self, int index);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveSubSkin(fgSkin* self, FG_UINT subskin);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSubSkin(fgSkin* self, FG_UINT subskin);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_AddSkin(fgSkin* self, char* name);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, char* name);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSkin(fgSkin* self, char* name);

FG_EXTERN void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, char* name, fgElement* element, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self);

FG_EXTERN void FG_FASTCALL fgStyle_Init(fgStyle* self);
FG_EXTERN void FG_FASTCALL fgStyle_Destroy(fgStyle* self);
FG_EXTERN FG_UINT FG_FASTCALL fgStyle_AddSubstyle(fgStyle* self, ptrdiff_t index);
FG_EXTERN char FG_FASTCALL fgStyle_RemoveSubstyle(fgStyle* self, FG_UINT substyle);
FG_EXTERN fgStyle* FG_FASTCALL fgStyle_GetSubstyle(fgStyle* self, FG_UINT substyle);

FG_EXTERN fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2);
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg);

FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadFileUBJSON(struct __kh_fgSkins_t* self, const char* file);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadUBJSON(struct __kh_fgSkins_t* self, const void* data, FG_UINT length);

#ifdef  __cplusplus
}
#endif

#endif