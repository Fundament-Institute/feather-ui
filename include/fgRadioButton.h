// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_RADIOBUTTON_H__
#define __FG_RADIOBUTTON_H__

#include "fgCheckbox.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A radio button is like a checkbox, but disables all other radio buttons that have the same parent control (use a non-styled fgElement to group them).
typedef struct _FG_RADIOBUTTON {
  fgCheckbox window; // A radio buton must be a different class for styling purposes.
  struct _FG_RADIOBUTTON* radionext; // Used for the list of radiobuttons in a given fgElement grouping
  struct _FG_RADIOBUTTON* radioprev;
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgRadiobutton;

FG_EXTERN void FG_FASTCALL fgRadiobutton_Init(fgRadiobutton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgRadiobutton_Destroy(fgRadiobutton* self);
FG_EXTERN size_t FG_FASTCALL fgRadiobutton_Message(fgRadiobutton* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif



#endif