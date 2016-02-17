// Copyright ©2016 Black Sphere Studios
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
typedef struct __FG_WINDOW {
  fgChild element;
  struct FG_MENU* contextmenu;
  char* name; // Optional name used for mapping to skin collections
  struct __FG_WINDOW* tabnext;
  struct __FG_WINDOW* tabprev;
} fgWindow;

FG_EXTERN fgChild* fgFocusedWindow;
FG_EXTERN fgChild* fgLastHover; // Last window the mouse moved over, used to generate MOUSEON and MOUSEOFF events
FG_EXTERN fgChild* fgCaptureWindow;

FG_EXTERN void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgWindow_Destroy(fgWindow* self);
FG_EXTERN size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgWindow_TabAfter(fgWindow* self, fgWindow* prev);
FG_EXTERN void FG_FASTCALL fgWindow_TabBefore(fgWindow* self, fgWindow* next);
FG_EXTERN void FG_FASTCALL fgWindow_DoHoverCalc(fgWindow* self);

#ifdef  __cplusplus
}
#endif

#endif
