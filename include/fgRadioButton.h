// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RADIOBUTTON_H__
#define __FG_RADIOBUTTON_H__

#include "fgCheckbox.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A radio button is like a checkbox, but disables all other radio buttons that have the same parent control (use a non-styled fgChild to group them).
typedef struct _FG_RADIOBUTTON {
  fgCheckbox window; // A radio buton must be a different class for styling purposes.
  struct _FG_RADIOBUTTON* radionext; // Used for the list of radiobuttons in a given fgChild grouping
  struct _FG_RADIOBUTTON* radioprev;
} fgRadiobutton;

FG_EXTERN fgChild* FG_FASTCALL fgRadiobutton_Create(const char* text, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgRadiobutton_Destroy(fgRadiobutton* self);
FG_EXTERN size_t FG_FASTCALL fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif