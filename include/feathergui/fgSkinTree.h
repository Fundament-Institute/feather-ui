// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SKIN_TREE_H__
#define __FG_SKIN_TREE_H__

#include "fgStyle.h"
#include "fgElement.h"

struct _FG_SKIN;

typedef struct _FG_SKIN_ELEMENT {
  const char* type;
  fgTransform transform;
  short units;
  fgFlag flags;
  fgStyle style; // style overrides
  struct _FG_SKIN* skin; // Skin override (cannot be passed as a message because this blows up absolutely everything)
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
  fgSkinElement element;
  fgSkinTree tree;
  fgElement* instance; // Instance of this skin element. Skin elements cannot be assigned skins, so this only needs to apply the style overrides.
} fgSkinLayout;

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

#endif