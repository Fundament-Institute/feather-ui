// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_BUTTON_H__
#define __FG_BUTTON_H__

#include "fgText.h"
#include "fgWindow.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGBUTTON_FLAGS
{
  FGBUTTON_NOFOCUS = (1 << 5),
  FGBUTTON_TOGGLE = (1 << 6),
};

// A button is usually implemented as a simple background design that takes a static and displays it in the center of the button.
typedef struct _FG_BUTTON {
  fgWindow window;
  fgChild item; // item displayed in button
  fgText text; // text displayed in button
  char state; // tracks if the button is down.
} fgButton;

FG_EXTERN fgChild* FG_FASTCALL fgButton_Create(fgChild* item, fgFlag flags, fgChild* parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgButton_Destroy(fgButton* self);
FG_EXTERN size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
