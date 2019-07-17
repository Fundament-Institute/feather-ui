// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__SKIN_H
#define FG__SKIN_H

#include "khash.h"
#include "outline.h"
#include "calc.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct
{
  fgCalcNode* nodes;
  unsigned int count;
} fgCalculation;

EXPORT__KHASH_PROTOTYPES(field, unsigned int, fgCalculation);
EXPORT__KHASH_PROTOTYPES(unresolved, const char*, fgCalculation);

// This is a runtime representation of a layout. Terra would compile this into a delegate at compile time.
typedef struct FG__SKIN_NODE
{
  const char* id;
  kh_unresolved_t* unresolved;
  kh_field_t* fields; // If this is a data expansion node, fields will be NULL and the data pointer will be evaluated directly
  void* data;
  struct FG__SKIN_NODE* children;
  struct FG__SKIN_NODE* sibling;
} fgSkinNode;

EXPORT__KHASH_PROTOTYPES(outline, const char*, struct FG__SKIN_NODE*);

// A resolved skin, where all data has been type-checked already.
typedef struct
{
  kh_outline_t* generators;
  struct FG__SKIN_NODE* root;
} fgSkin;

// This is a meta-layer above the outlines which processes the data bindings for a skin.
FG_COMPILER_DLLEXPORT fgOutlineNode* fgSkinGenerator(struct FG__ROOT* root, fgSkin* skin, float scale, fgVec dpi);
FG_COMPILER_DLLEXPORT void fgSkinResolveFields(kh_field_t* fields, kh_unresolved_t* unresolved, fgResolver resolver);

#ifdef  __cplusplus
}
#endif

#endif
