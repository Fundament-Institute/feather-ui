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
  ptrdiff_t index; // Index of the child this applies to, if applicable. Defaults to 0. Negative numbers are valid but map to custom statics, so they must be applied manually by the control.
  FG_Msg msg;
  struct __FG_STYLE_MSG* next;
} fgStyleMsg;

typedef struct __FG_STYLE
{
  fgStyleMsg* styles;
  fgVector substyles; // type: fgStyle
} fgStyle;

typedef struct __FG_FULL_DEF
{
  void* def;
  fgElement element;
  int order;
} fgFullDef;

typedef struct __FG_SKIN
{
  fgVector statics; // type: fgFullDef
  fgVector styles; // type: fgStyle
  fgVector subskins; // type: fgSkin
} fgSkin;

struct __kh_fgSkins_t;

FG_EXTERN void FG_FASTCALL fgSkin_Init(fgSkin* self);
FG_EXTERN void FG_FASTCALL fgSkin_Destroy(fgSkin* self);
FG_EXTERN FG_UINT FG_FASTCALL fgStyle_AddStyle(fgVector* self);
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyle(fgVector* self, FG_UINT index);
FG_EXTERN fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2);
FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddStatic(fgSkin* self, void* def, const fgElement* elem, int order);
FG_EXTERN void FG_FASTCALL fgSkin_RemoveStatic(fgSkin* self, FG_UINT index);
FG_EXTERN FG_UINT FG_FASTCALL fgSkin_AddSkin(fgSkin* self);
FG_EXTERN fgSkin* FG_FASTCALL fgSkin_GetSkin(fgSkin* self, FG_UINT index);
FG_EXTERN void FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, FG_UINT index);
FG_EXTERN void FG_FASTCALL fgSkin_RemoveStatic(fgSkin* self, FG_UINT index);
#define fgSkin_GetStyle(self, i) fgVector_GetP(self->styles, i, fgStyle)
FG_EXTERN struct __kh_fgSkins_t* FG_FASTCALL fgSkins_Init();
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_Add(struct __kh_fgSkins_t* self, const char* name);
FG_EXTERN fgSkin* FG_FASTCALL fgSkins_Get(struct __kh_fgSkins_t* self, const char* name);
FG_EXTERN void FG_FASTCALL fgSkins_Remove(struct __kh_fgSkins_t* self, const char* name);
FG_EXTERN void FG_FASTCALL fgSkins_Destroy(struct __kh_fgSkins_t* self);
FG_EXTERN void FG_FASTCALL fgSkins_Apply(struct __kh_fgSkins_t* self, fgWindow* window);

#ifdef  __cplusplus
}
#endif

#endif