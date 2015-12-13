// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_CHECKBOX_H__
#define __FG_CHECKBOX_H__

#include "fgWindow.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A checkbox is a toggleable control with text alongside it.
typedef struct _FG_CHECKBOX {
  fgWindow window;
  fgChild check; // Displayed when the checkbox is checked.
  fgChild indeterminate; // Displayed when the checkbox state is set to 2 (indeterminate state)
  fgChild item; // item displayed
  fgText text; // text displayed
  char checked;
} fgCheckbox;

FG_EXTERN fgChild* FG_FASTCALL fgCheckbox_Create(const char* text, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgCheckbox_Destroy(fgCheckbox* self);
FG_EXTERN size_t FG_FASTCALL fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif