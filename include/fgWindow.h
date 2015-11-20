// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_WINDOW_H__
#define __FG_WINDOW_H__

#include "fgChild.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct FG_MENU;
struct __FG_SKIN;

// Defines the base GUI element, a window. This is not an actual top level window.
typedef struct __WINDOW {
  fgChild element;
  FG_UINT id;
  struct FG_MENU* contextmenu;
  char* name; // Optional name used for mapping to skin collections
} fgWindow;

FG_EXTERN fgWindow* fgFocusedWindow;
FG_EXTERN fgWindow* fgLastHover; // Last window the mouse moved over, used to generate MOUSEON and MOUSEOFF events

FG_EXTERN void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgWindow_Destroy(fgWindow* self);
FG_EXTERN size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
