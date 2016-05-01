// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_CHECKBOX_H__
#define _FG_CHECKBOX_H__

#include "fgControl.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A checkbox is a toggleable control with text alongside it.
typedef struct _FG_CHECKBOX {
  fgControl control;
  fgElement check; // Displayed when the checkbox is checked.
  fgElement indeterminate; // Displayed when the checkbox state is set to 2 (indeterminate state)
  fgElement item; // item displayed
  fgText text; // text displayed
  char checked;
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgCheckbox;

FG_EXTERN fgElement* FG_FASTCALL fgCheckbox_Create(const char* text, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgCheckbox_Destroy(fgCheckbox* self);
FG_EXTERN size_t FG_FASTCALL fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif