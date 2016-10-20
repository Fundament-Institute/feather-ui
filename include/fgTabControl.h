// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TABCONTROL_H__
#define __FG_TABCONTROL_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A Tab group is a collection of tab pages that can be switched between by clicking on "tabs".
typedef struct {
  fgControl control; // use FG_ADDITEM to add a panel with a name to the tabs. It returns a pointer to the added element you can then use with REMOVEITEM.
  fgBox header; // This is where the radiobuttons are added, allowing the "fgRadioButton" class to be skinned. Uses FG_BACKGROUND so padding can be applied to the panels.
  fgElement* selected; // This is the currently active tab panel
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
  FG_DLLEXPORT fgElement* AddItem(const char* name);
#endif
} fgTabControl;

FG_EXTERN void FG_FASTCALL fgTabControl_Init(fgTabControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgTabControl_Destroy(fgTabControl* self);
FG_EXTERN size_t FG_FASTCALL fgTabControl_Message(fgTabControl* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif