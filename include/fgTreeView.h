// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TREEVIEW_H__
#define __FG_TREEVIEW_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct _FG_TREEITEM {
  fgControl control;
  fgElement arrow;
  size_t count;
} fgTreeItem;

// A treeview visualizes a tree structure as a series of nested lists. 
typedef struct _FG_TREEVIEW {
  fgScrollbar scrollbar;
#ifdef  __cplusplus
  inline operator fgElement*() { return &scrollbar.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTreeview;

FG_EXTERN void FG_FASTCALL fgTreeview_Init(fgTreeview* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgTreeview_Destroy(fgTreeview* self);
FG_EXTERN size_t FG_FASTCALL fgTreeview_Message(fgTreeview* self, const FG_Msg* msg);

FG_EXTERN void FG_FASTCALL fgTreeItem_Init(fgTreeItem* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN size_t FG_FASTCALL fgTreeItem_Message(fgTreeItem* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgTreeItem_Destroy(fgTreeItem* self);

#ifdef  __cplusplus
}
#endif

#endif