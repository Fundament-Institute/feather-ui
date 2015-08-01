// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_WINDOW_H__
#define __FG_WINDOW_H__

#include "fgStatic.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FG_WINFLAGS
{
  FGWIN_NOCLIP=1,
  FGWIN_HIDDEN=2,
  FGGRID_EXPANDX=4,
  FGGRID_EXPANDY=8,
  FGGRID_TILEX=16,
  FGGRID_TILEY=32,
  FGGRID_FIXEDX = 64,
  FGGRID_FIXEDY = 128,
  FGSCROLLBAR_VERT=16,
  FGSCROLLBAR_HORZ=32,
  FGSCROLLBAR_HIDEX=64,
  FGSCROLLBAR_HIDEY=128,
  FGTOPWINDOW_MINIMIZE=16,
  FGTOPWINDOW_RESTORE=32,
  FGTOPWINDOW_RESIZABLE=64,
  FGTOPWINDOW_NOTITLEBAR=128,
  FGTOPWINDOW_NOBORDER=256,
};

struct FG_MENU;
struct __FG_SKIN;

// Defines the base GUI element, a window. This is not an actual top level window.
typedef struct __WINDOW {
  fgChild element;
  char (FG_FASTCALL *message)(void* self, const FG_Msg* msg);
  FG_UINT id;
  fgStatic* rlist; // root node for statics
  fgStatic* rlast; // last node for statics 
  struct FG_MENU* contextmenu;
  const struct __FG_SKIN* skin; // skin reference
  fgVector skinstatics; // array of statics that belong to the skin and are referenced by it
  char* name; // Optional name used for mapping to skin collections
  void* userdata;
} fgWindow;

FG_EXTERN fgWindow* fgFocusedWindow;
FG_EXTERN fgWindow* fgLastHover; // Last window the mouse moved over, used to generate MOUSEON and MOUSEOFF events

FG_EXTERN void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgWindow* BSS_RESTRICT parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgWindow_Destroy(fgWindow* self);
FG_EXTERN char FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgWindow_SetElement(fgWindow* self, const fgElement* element);
FG_EXTERN void FG_FASTCALL fgWindow_SetArea(fgWindow* self, const CRect* area);
FG_EXTERN char FG_FASTCALL fgWindow_BasicMessage(fgWindow* self, unsigned char type); // Shortcut for sending type messages with no data
FG_EXTERN char FG_FASTCALL fgWindow_VoidMessage(fgWindow* self, unsigned char type, void* data); // Shortcut for sending void* messages
FG_EXTERN char FG_FASTCALL fgWindow_VoidAuxMessage(fgWindow* self, unsigned char type, void* data, int aux);
FG_EXTERN char FG_FASTCALL fgWindow_IntMessage(fgWindow* self, unsigned char type, int data); // Shortcut for sending int messages
FG_EXTERN void FG_FASTCALL fgWindow_SetParent(fgWindow* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent);
FG_EXTERN char FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgWindow_TriggerStyle(fgWindow* self, FG_UINT index);

#ifdef  __cplusplus
}
#endif

#endif
