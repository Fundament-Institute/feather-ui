// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_COMBOBOX_H__
#define __FG_COMBOBOX_H__

#include "fgTextbox.h"
#include "fgDropdown.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A combobox is a text-only dropdown that allows the user to edit the text
typedef struct {
  fgDropdown dropdown; // GETSELECTEDITEM returns either the selected item or the text typed into the combobox.
  fgTextbox text;
#ifdef  __cplusplus
  inline operator fgElement*() { return &dropdown.box.scroll.control.element; }
#endif
} fgCombobox;

FG_EXTERN void fgCombobox_Init(fgCombobox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgCombobox_Destroy(fgCombobox* self);
FG_EXTERN size_t fgCombobox_Message(fgCombobox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
