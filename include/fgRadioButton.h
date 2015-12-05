// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RADIOBUTTON_H__
#define __FG_RADIOBUTTON_H__

#include "fgCheckbox.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A radio button is like a checkbox, but disables all other radio buttons that have the same parent control (use a non-styled fgChild to group them).
typedef struct _FG_RADIOBUTTON {
  fgCheckbox window; // While a radio button almost identical to a checkbox, it must be a different class for styling purposes.
} fgRadiobutton;

FG_EXTERN fgChild* FG_FASTCALL fgRadiobutton_Create(fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgRadiobutton_Destroy(fgRadiobutton* self);
FG_EXTERN size_t FG_FASTCALL fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif