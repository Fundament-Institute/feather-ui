// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_COMBOBOX_H__
#define _FG_COMBOBOX_H__

#include "fgTextbox.h"
#include "fgButton.h"
#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGCOMBOBOX_FLAGS
{
  FGCOMBOBOX_EDITBOX = (FGSCROLLBAR_SHOWV << 1),
};

// A combobox is a text-only dropdown that can optionally let the user edit the text.
typedef struct {
  fgTextbox box; // GETSELECTEDITEM returns either the selected item or the text typed into the combobox.
  fgButton button;
  fgBox dropdown; // ADDITEM is used to add an arbitrary object to the list. 
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.window.control.element; }
#endif
} fgCombobox;

FG_EXTERN void FG_FASTCALL fgCombobox_Init(fgCombobox* self, fgControl* parent, fgFlag flags, const fgTransform* transform);
FG_EXTERN char FG_FASTCALL fgCombobox_Message(fgCombobox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
