// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_COMBOBOX_H__
#define __FG_COMBOBOX_H__

#include "fgTextbox.h"
#include "fgButton.h"
#include "fgList.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGCOMBOBOX_FLAGS
{
  FGCOMBOBOX_EDITBOX = (1 << 16),
};

// A combobox is a text-only dropdown that can optionally let the user edit the text.
typedef struct {
  fgTextbox box; // GETSELECTEDITEM returns either the selected item or the text typed into the combobox.
  fgButton button;
  fgList dropdown; // ADDITEM is used to add an arbitrary object to the list. 
} fgCombobox;

FG_EXTERN fgWindow* FG_FASTCALL fgCombobox_Create(fgWindow* parent, const fgElement* element, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgCombobox_Init(fgCombobox* self, fgWindow* parent, const fgElement* element, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgCombobox_Message(fgCombobox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
