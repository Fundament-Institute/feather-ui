// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_MENU_H__
#define __FG_MENU_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_MENU;

typedef struct _FG_MENUITEM {
  fgElement element;
  fgText text;
  struct _FG_MENU* submenu;
} fgMenuItem;

// A Menu is either a window menu or a context menu. Turns into a menubar if made the child of a top-level window
typedef struct _FG_MENU {
  fgBox box;
  fgElement arrow;
  struct _FG_MENU* expanded; // holds the submenu that is currently expanded, if one is.
  //fgDeferAction* dropdown; // Keeps track of our dropdown action in fgRoot
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.scroll.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgMenu;

FG_EXTERN void fgMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void fgMenu_Destroy(fgMenu* self);
FG_EXTERN size_t fgMenu_Message(fgMenu* self, const FG_Msg* msg);
//FG_EXTERN char fgMenu_DoDropdown(fgMenu* self);
FG_EXTERN void fgSubmenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN size_t fgSubmenu_Message(fgMenu* self, const FG_Msg* msg);
FG_EXTERN void fgMenuItem_Init(fgMenuItem* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN size_t fgMenuItem_Message(fgMenuItem* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif
