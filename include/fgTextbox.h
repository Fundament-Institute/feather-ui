// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TEXTBOX_H__
#define __FG_TEXTBOX_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A Textbox is really just a text static inside an optional Scrollbar. It can be single or multi-line.
typedef struct {
  fgScrollbar window;
  fgChild text;
} fgTextbox;

FG_EXTERN fgWindow* FG_FASTCALL fgTextbox_Create(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgTextbox_Init(fgTextbox* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgTextbox_Message(fgTextbox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif