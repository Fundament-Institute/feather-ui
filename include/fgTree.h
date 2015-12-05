// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TREE_H__
#define __FG_TREE_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A treeview visualizes a tree structure as a series of nested lists
typedef struct _FG_TREE {
  fgScrollbar window;
  fgChild expand;
  fgChild shrink;
} fgTree;

FG_EXTERN fgChild* FG_FASTCALL fgTree_Create(fgChild* item, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgTree_Init(fgTree* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgTree_Destroy(fgTree* self);
FG_EXTERN size_t FG_FASTCALL fgTree_Message(fgTree* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif