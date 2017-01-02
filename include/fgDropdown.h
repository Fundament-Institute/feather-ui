// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_DROPDOWN_H__
#define __FG_DROPDOWN_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A dropdown contains a list of arbitrary items and allows the user to select one.
typedef struct {
  fgControl control;
  fgBox box;
  fgElement* selected;
  fgColor hover;
  fgColor select;
  char dropflag;
  fgMouseState mouse;
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgDropdown;

FG_EXTERN void FG_FASTCALL fgDropdown_Init(fgDropdown* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgDropdown_Destroy(fgDropdown* self);
FG_EXTERN size_t FG_FASTCALL fgDropdown_Message(fgDropdown* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
