// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_DROPDOWN_H__
#define __FG_DROPDOWN_H__

#include "fgList.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A dropdown contains a list of arbitrary items and allows the user to select one.
typedef struct {
  fgList window; // ADDITEM is used to add an arbitrary object to the list. 
} fgDropdown;

FG_EXTERN fgWindow* FG_FASTCALL fgDropdown_Create(fgWindow* parent, const fgElement* element, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgDropdown_Init(fgDropdown* self, fgWindow* parent, const fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgDropdown_Message(fgDropdown* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
