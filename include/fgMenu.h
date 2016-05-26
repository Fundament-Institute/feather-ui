// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_MENU_H__
#define _FG_MENU_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_MENU;
typedef struct _FG_SUBMENU {
  fgElement element;
  struct _FG_MENU* submenu;
} fgSubMenu;

// A Menu is either a window menu or a context menu. Turns into a menubar if made the child of a top-level window
typedef struct _FG_MENU {
  fgScrollbar window;
  fgElement highlight;
  fgElement arrow;
  fgElement seperator; // cloned to replace a null value inserted into the list. This allows a style to control the size and appearence of the seperator.
  struct _FG_MENU* expanded; // holds the submenu that is currently expanded, if one is.
  //fgDeferAction* dropdown; // Keeps track of our dropdown action in fgRoot
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgMenu;

FG_EXTERN void FG_FASTCALL fgMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, char submenu);
FG_EXTERN void FG_FASTCALL fgMenu_Destroy(fgMenu* self);
FG_EXTERN size_t FG_FASTCALL fgMenu_Message(fgMenu* self, const FG_Msg* msg);
FG_EXTERN size_t FG_FASTCALL fgSubmenu_Message(fgMenu* self, const FG_Msg* msg);
//FG_EXTERN char FG_FASTCALL fgMenu_DoDropdown(fgMenu* self);

#ifdef  __cplusplus
}
#endif

#endif
