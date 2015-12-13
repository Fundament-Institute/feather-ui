// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TREEVIEW_H__
#define __FG_TREEVIEW_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A treeview visualizes a tree structure as a series of nested lists
typedef struct _FG_TREEVIEW {
  fgScrollbar window;
  fgChild expand;
  fgChild shrink;
} fgTreeView;

FG_EXTERN fgChild* FG_FASTCALL fgTreeView_Create(fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgTreeView_Init(fgTreeView* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgTreeView_Destroy(fgTreeView* self);
FG_EXTERN size_t FG_FASTCALL fgTreeView_Message(fgTreeView* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif