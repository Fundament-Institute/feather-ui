// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SKIN_H__
#define __FG_SKIN_H__

#include "fgStyle.h"
#include "fgControl.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _FG_STYLE_LAYOUT {
  char* type;
  char* name;
  char* id;
  fgTransform transform;
  short units;
  fgFlag flags;
  fgStyle style; // style overrides
  int order;
} fgStyleLayout;

struct __kh_fgSkins_t;
struct __kh_fgStyles_t;
struct __kh_fgStyleInt_t;

typedef struct _FG_SKIN_BASE
{
  fgVector resources; // type: void*
  fgVector fonts;
  struct __kh_fgSkins_t* skinmap;
  struct _FG_SKIN_BASE* parent;
} fgSkinBase;

typedef struct _FG_SKIN
{
  struct _FG_SKIN_BASE base;
  struct _FG_SKIN* inherit;
  fgStyle style; // style overrides
  fgDeclareVector(fgStyleLayout, StyleLayout) children; // type: fgStyleLayout
  struct __kh_fgStyleInt_t* styles;
} fgSkin;


FG_EXTERN void FG_FASTCALL fgSkin_Init(fgSkin* self);
FG_EXTERN void FG_FASTCALL fgSkin_Destroy(fgSkin* self);
FG_EXTERN void FG_FASTCALL fgSkinBase_Destroy(fgSkinBase* self);
FG_EXTERN size_t FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child);
FG_EXTERN fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self, const char* name);
FG_EXTERN char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style);
FG_EXTERN fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name);

FG_EXTERN fgSkin* FG_FASTCALL fgSkinBase_AddSkin(fgSkinBase* self, const char* name);
FG_EXTERN char FG_FASTCALL fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name);
FG_EXTERN fgSkin* FG_FASTCALL fgSkinBase_GetSkin(const fgSkinBase* self, const char* name);
FG_EXTERN size_t FG_FASTCALL fgSkinBase_AddResource(fgSkinBase* self, void* resource);
FG_EXTERN char FG_FASTCALL fgSkinBase_RemoveResource(fgSkinBase* self, FG_UINT resource);
FG_EXTERN void* FG_FASTCALL fgSkinBase_GetResource(const fgSkinBase* self, FG_UINT resource);
FG_EXTERN size_t FG_FASTCALL fgSkinBase_AddFont(fgSkinBase* self, void* font);
FG_EXTERN char FG_FASTCALL fgSkinBase_RemoveFont(fgSkinBase* self, FG_UINT font);
FG_EXTERN void* FG_FASTCALL fgSkinBase_GetFont(const fgSkinBase* self, FG_UINT font);
FG_EXTERN fgFlag fgSkinBase_GetFlagsFromString(const char* s, fgFlag* remove);

FG_EXTERN void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self);

FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadFileUBJSON(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadUBJSON(fgSkinBase* self, const void* data, FG_UINT length);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadFileXML(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_LoadXML(fgSkinBase* self, const char* data, FG_UINT length);

#ifdef  __cplusplus
}
#endif

#endif