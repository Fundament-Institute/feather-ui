// Copyright ©2016 Black Sphere Studios
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
  fgDropdown box; // GETSELECTEDITEM returns either the selected item or the text typed into the combobox.
  fgTextbox text;
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.box.scroll.control.element; }
#endif
} fgCombobox;

FG_EXTERN void FG_FASTCALL fgCombobox_Init(fgCombobox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgCombobox_Destroy(fgCombobox* self);
FG_EXTERN size_t FG_FASTCALL fgCombobox_Message(fgCombobox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
