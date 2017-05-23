// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_WINDOW_H__
#define __FG_WINDOW_H__

#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGWINDOW_FLAGS
{
  FGWINDOW_MINIMIZABLE = (FGCONTROL_DISABLE << 1),
  FGWINDOW_MAXIMIZABLE = (FGWINDOW_MINIMIZABLE << 1),
  FGWINDOW_RESIZABLE = (FGWINDOW_MAXIMIZABLE << 1),
  FGWINDOW_NOTITLEBAR = (FGWINDOW_RESIZABLE << 1),
  FGWINDOW_NOBORDER = (FGWINDOW_NOTITLEBAR << 1),
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
  char maximized; // 1 if maximized
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgWindow;

FG_EXTERN void fgWindow_Init(fgWindow* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
FG_EXTERN void fgWindow_Destroy(fgWindow* self);
FG_EXTERN size_t fgWindow_Message(fgWindow* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
