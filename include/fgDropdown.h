// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_DROPDOWN_H__
#define _FG_DROPDOWN_H__

#include "fgList.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A dropdown contains a list of arbitrary items and allows the user to select one.
typedef struct {
  fgList window; // ADDITEM is used to add an arbitrary object to the list. 
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgDropdown;

FG_EXTERN fgControl* FG_FASTCALL fgDropdown_Create(fgControl* parent, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgDropdown_Init(fgDropdown* self, fgControl* parent, fgFlag flags, const fgTransform* transform);
FG_EXTERN char FG_FASTCALL fgDropdown_Message(fgDropdown* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
