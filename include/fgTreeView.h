// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TREEVIEW_H__
#define _FG_TREEVIEW_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _FG_TREEVIEW_CONTAINER {
  fgElement window;
  fgElement* content;
} fgTreeViewContainer;

// A treeview visualizes a tree structure as a series of nested lists. 
typedef struct _FG_TREEVIEW {
  fgScrollbar window;
  fgElement expand;
  fgElement shrink;
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTreeView;

FG_EXTERN fgElement* FG_FASTCALL fgTreeView_Create(fgFlag flags, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT prev, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgTreeView_Init(fgTreeView* BSS_RESTRICT self, fgFlag flags, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT prev, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgTreeView_Destroy(fgTreeView* self);
FG_EXTERN size_t FG_FASTCALL fgTreeView_Message(fgTreeView* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgTreeViewContainer_Message(fgTreeViewContainer* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif