// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_WINDOW_H__
#define _FG_WINDOW_H__

#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGWINDOW_FLAGS
{
  FGWINDOW_MINIMIZABLE = (1 << 8),
  FGWINDOW_MAXIMIZABLE = (1 << 9),
  FGWINDOW_RESIZABLE = (1 << 10),
  FGWINDOW_NOTITLEBAR = (1 << 11),
  FGWINDOW_NOBORDER = (1 << 12),
};

enum FGWINDOW_ACTIONS
{
  FGWINDOW_CLOSE = 0,
  FGWINDOW_MAXIMIZE,
  FGWINDOW_RESTORE,
  FGWINDOW_MINIMIZE,
  FGWINDOW_UNMINIMIZE,
};

struct _FG_BUTTON;

// A top-level window is an actual window with a titlebar that can be dragged and resized.
typedef struct _FG_WINDOW {
  fgControl control;
  fgText caption;
  fgButton controls[3]; // 0 is the close button, 1 is the maximize/restore button, 2 is the minimize button
  CRect prevrect; // Stores where the window was before being maximized
  AbsVec offset; // offset from the mouse cursor for movement
  char dragged; // 1 if currently being dragged by mouse
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgWindow;

FG_EXTERN fgElement* FG_FASTCALL fgWindow_Create(const char* caption, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgWindow_Init(fgWindow* self, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgWindow_Destroy(fgWindow* self);
FG_EXTERN size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
