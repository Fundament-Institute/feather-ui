// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SKIN_H__
#define __FG_SKIN_H__

#include "fgStyle.h"
#include "fgControl.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _FG_SKIN_ELEMENT {
  const char* type;
  fgTransform transform;
  short units;
  fgFlag flags;
  fgStyle style; // style overrides
  int order;
} fgSkinElement;

struct _FG_SKIN_LAYOUT;
typedef fgDeclareVector(struct _FG_SKIN_LAYOUT, SkinLayout) fgVectorSkinLayout;

typedef struct _FG_SKIN_TREE {
  struct fgStylePair {
    FG_UINT map;
    fgStyle style;
  };

  fgVectorSkinLayout children;
  fgDeclareVector(struct fgStylePair, StylePair) styles;
  size_t stylemask;

#ifdef  __cplusplus
  FG_DLLEXPORT size_t AddChild(const char* type, fgFlag flags, const fgTransform* transform, short units, int order);
  FG_DLLEXPORT char RemoveChild(FG_UINT child);
  FG_DLLEXPORT struct _FG_SKIN_LAYOUT* GetChild(FG_UINT child) const;
  FG_DLLEXPORT FG_UINT AddStyle(const char* name);
  FG_DLLEXPORT char RemoveStyle(FG_UINT style);
  FG_DLLEXPORT fgStyle* GetStyle(FG_UINT style) const;
#endif
} fgSkinTree;

typedef struct _FG_SKIN_LAYOUT {
  fgSkinElement layout;
  fgSkinTree tree;
  fgElement* instance; // Instance of this skin element. Skin elements cannot be assigned skins, so this only needs to apply the style overrides.
} fgSkinLayout;

struct __kh_fgSkins_t;
struct __kh_fgStyles_t;
struct _FG_SKIN;

typedef struct _FG_SKIN_BASE
{
  fgVector resources; // type: void*
  fgVector fonts;
  struct __kh_fgSkins_t* skinmap;
  struct _FG_SKIN_BASE* parent;

#ifdef  __cplusplus
  FG_DLLEXPORT struct _FG_SKIN* AddSkin(const char* name);
  FG_DLLEXPORT char RemoveSkin(const char* name);
  FG_DLLEXPORT struct _FG_SKIN* GetSkin(const char* name) const;
  FG_DLLEXPORT size_t AddAsset(void* resource);
  FG_DLLEXPORT char RemoveAsset(FG_UINT resource);
  FG_DLLEXPORT void* GetAsset(FG_UINT resource) const;
  FG_DLLEXPORT size_t AddFont(void* font);
  FG_DLLEXPORT char RemoveFont(FG_UINT font);
  FG_DLLEXPORT void* GetFont(FG_UINT font) const;

  FG_DLLEXPORT struct _FG_SKIN* LoadFileUBJSON(const char* file);
  FG_DLLEXPORT struct _FG_SKIN* LoadUBJSON(const void* data, FG_UINT length);
  FG_DLLEXPORT struct _FG_SKIN* LoadFileXML(const char* file);
  FG_DLLEXPORT struct _FG_SKIN* LoadXML(const char* data, FG_UINT length);
#endif
} fgSkinBase;

typedef struct _FG_SKIN
{
  struct _FG_SKIN_BASE base;
  struct _FG_SKIN* inherit;
  fgStyle style; // style overrides
  fgSkinTree tree;

#ifdef  __cplusplus
  FG_DLLEXPORT struct _FG_SKIN* GetSkin(const char* name) const;
#endif
} fgSkin;

FG_EXTERN void fgSkinElement_Init(fgSkinElement* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN void fgSkinElement_Destroy(fgSkinElement* self);

FG_EXTERN void fgSkinTree_Init(fgSkinTree* self);
FG_EXTERN void fgSkinTree_Destroy(fgSkinTree* self);
FG_EXTERN size_t fgSkinTree_AddChild(fgSkinTree* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN char fgSkinTree_RemoveChild(fgSkinTree* self, FG_UINT child);
FG_EXTERN fgSkinLayout* fgSkinTree_GetChild(const fgSkinTree* self, FG_UINT child);
FG_EXTERN FG_UINT fgSkinTree_AddStyle(fgSkinTree* self, const char* name);
FG_EXTERN char fgSkinTree_RemoveStyle(fgSkinTree* self, FG_UINT style);
FG_EXTERN fgStyle* fgSkinTree_GetStyle(const fgSkinTree* self, FG_UINT style);

FG_EXTERN void fgSkinLayout_Init(fgSkinLayout* self, const char* type, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN void fgSkinLayout_Destroy(fgSkinLayout* self);

FG_EXTERN void fgSkin_Init(fgSkin* self);
FG_EXTERN void fgSkin_Destroy(fgSkin* self);
FG_EXTERN fgSkin* fgSkin_GetSkin(const fgSkin* self, const char* name);
FG_EXTERN const char* fgSkin_ParseFont(const char* s, char quote, int* size, short* weight, char* italic);
FG_EXTERN char* fgSkin_GetFontFamily(char* s, char quote, char** context);

FG_EXTERN void fgSkinBase_Destroy(fgSkinBase* self);
FG_EXTERN fgSkin* fgSkinBase_AddSkin(fgSkinBase* self, const char* name);
FG_EXTERN char fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name);
FG_EXTERN fgSkin* fgSkinBase_GetSkin(const fgSkinBase* self, const char* name);
FG_EXTERN size_t fgSkinBase_AddAsset(fgSkinBase* self, void* asset);
FG_EXTERN char fgSkinBase_RemoveAsset(fgSkinBase* self, FG_UINT asset);
FG_EXTERN void* fgSkinBase_GetAsset(const fgSkinBase* self, FG_UINT asset);
FG_EXTERN size_t fgSkinBase_AddFont(fgSkinBase* self, void* font);
FG_EXTERN char fgSkinBase_RemoveFont(fgSkinBase* self, FG_UINT font);
FG_EXTERN void* fgSkinBase_GetFont(const fgSkinBase* self, FG_UINT font);
FG_EXTERN fgFlag fgSkinBase_GetFlagsFromString(const char* s, fgFlag* remove);

FG_EXTERN fgSkin* fgSkinBase_LoadFileUBJSON(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* fgSkinBase_LoadUBJSON(fgSkinBase* self, const void* data, FG_UINT length);
FG_EXTERN fgSkin* fgSkinBase_LoadFileXML(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* fgSkinBase_LoadXML(fgSkinBase* self, const char* data, FG_UINT length);

#ifdef  __cplusplus
}
#endif

#endif