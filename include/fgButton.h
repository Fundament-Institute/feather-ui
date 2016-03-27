// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_BUTTON_H__
#define _FG_BUTTON_H__

#include "fgWindow.h"
#include "fgText.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGBUTTON_FLAGS
{
  FGBUTTON_NOFOCUS = (1 << 5),
};

// A button is usually implemented as a simple background design that takes a static and displays it in the center of the button.
typedef struct _FG_BUTTON {
  fgWindow window;
  fgChild item; // item displayed in button
  fgText text; // text displayed in button
#ifdef  __cplusplus
  inline operator fgChild*() { return &window.element; }
  inline fgChild* operator->() { return operator fgChild*(); }
#endif
} fgButton;

FG_EXTERN fgChild* FG_FASTCALL fgButton_Create(const char* text, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgButton_Destroy(fgButton* self);
FG_EXTERN size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
