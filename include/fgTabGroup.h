// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_TABGROUP_H__
#define _FG_TABGROUP_H__

#include "fgControl.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A Tab group is a collection of tab pages that can be switched between by clicking on "tabs".
typedef struct {
  fgControl control; // use FG_ADDITEM to add a panel and a name to the tabs. It returns a pointer to the added element you can then use with REMOVEITEM.
  fgElement header; // This is where the radiobuttons are added, allowing the "fgReadioButton" class to be skinned. Uses FG_BACKGROUND so padding can be applied to the panels.
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTabGroup;

FG_EXTERN fgControl* FG_FASTCALL fgTabGroup_Create(fgControl* parent, const fgTransform* transform, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgTabGroup_Init(fgTabGroup* self, fgControl* parent, const fgTransform* transform, FG_UINT id, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgTabGroup_Message(fgTabGroup* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif