// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_CHECKBOX_H__
#define __FG_CHECKBOX_H__

#include "fgControl.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A checkbox is a toggleable control with text alongside it.
typedef struct _FG_CHECKBOX {
  fgControl control;
  fgText text; // text displayed
  char checked;
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgCheckbox;

FG_EXTERN void fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgCheckbox_Destroy(fgCheckbox* self);
FG_EXTERN size_t fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif