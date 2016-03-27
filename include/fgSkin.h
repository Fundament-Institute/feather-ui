// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_SKIN_H__
#define _FG_SKIN_H__

#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _FG_STYLE_MSG
{
  FG_Msg msg;
  struct _FG_STYLE_MSG* next;
} fgStyleMsg;

typedef struct _FG_STYLE
{
  fgStyleMsg* styles;
} fgStyle;

typedef struct _FG_STYLE_LAYOUT {
  char* name;
  fgElement element;
  fgFlag flags;
  fgStyle style; // style overrides
} fgStyleLayout;

struct __kh_fgSkins_t;
struct __kh_fgStyles_t;

typedef struct _FG_SKIN
{
  ptrdiff_t index; // Index of the child this applies to (only used for subskins). Negative numbers map to existing children, equal to index + prechild.
  fgStyle style; // style overrides
  fgVector resources; // type: void*
  fgVector fonts;
  fgDeclareVector(fgStyleLayout, StyleLayout) children; // type: fgStyleLayout
  fgDeclareVector(fgStyle, Style) styles; // type: fgStyle
  fgDeclareVector(struct _FG_SKIN, Skin) subskins; // type: fgSkin
  struct __kh_fgSkins_t* skinmap;
} fgSkin;


FG_EXTERN void FG_FASTCALL fgSubskin_Init(fgSkin* self, int index);
FG_EXTERN void FG_FASTCALL fgSkin_Init(fgSkin* self);
FG_EXTERN void FG_FASTCALL fgSkin_Destroy(fgSkin* self);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddResource(fgSkin* self, void* resource);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveResource(fgSkin* self, FG_UINT resource);
FG_EXTERN void* FG_FASTCALL fgSkin_GetResource(const fgSkin* self, FG_UINT resource);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddFont(fgSkin* self, void* font);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveFont(fgSkin* self, FG_UINT font);
FG_EXTERN void* FG_FASTCALL fgSkin_GetFont(const fgSkin* self, FG_UINT font);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* name, const fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child);
FG_EXTERN fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddStyle(fgSkin* self, const char* name);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style);
FG_EXTERN fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddSubSkin(fgSkin* self, int index);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveSubSkin(fgSkin* self, FG_UINT subskin);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSubSkin(const fgSkin* self, FG_UINT subskin);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_AddSkin(fgSkin* self, const char* name);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, const char* name);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name);

FG_EXTERN void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* name, const fgElement* element, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self);

FG_EXTERN void FG_FASTCALL fgStyle_Init(fgStyle* self);
FG_EXTERN void FG_FASTCALL fgStyle_Destroy(fgStyle* self);
FG_EXTERN FG_UINT FG_FASTCALL fgStyle_GetName(const char* name);

FG_EXTERN fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2);
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg);

FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadFileUBJSON(struct __kh_fgSkins_t* self, const char* file);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadUBJSON(struct __kh_fgSkins_t* self, const void* data, FG_UINT length);

#ifdef  __cplusplus
}
#endif

#endif