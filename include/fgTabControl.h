// Copyright ©2017 Black Sphere Studios
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
  fgBox header; // This is where the radiobuttons are added, allowing the "fgRadiobutton" class to be skinned. Uses FG_BACKGROUND so padding can be applied to the panels.
  fgElement* selected; // This is the currently active tab panel
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgTabcontrol;

FG_EXTERN void fgTabcontrol_Init(fgTabcontrol* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgTabcontrol_Destroy(fgTabcontrol* self);
FG_EXTERN size_t fgTabcontrol_Message(fgTabcontrol* self, const FG_Msg* msg);

FG_EXTERN size_t fgTab_Message(fgElement* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif