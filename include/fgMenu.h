// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_MENU_H__
#define __FG_MENU_H__

#include "fgScrollbar.h"
#include "fgRoot.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A Menu is either a window menu or a context menu. Turns into a menubar if made the child of a top-level window
typedef struct FG_MENU {
  fgScrollbar window;
  fgChild* highlight;
  fgChild* seperator;
  fgChild* arrow;
  fgDeferAction* dropdown; // Keeps track of our dropdown action in fgRoot
} fgMenu;

FG_EXTERN fgChild* FG_FASTCALL fgMenu_Create(fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgMenu_Init(fgMenu* self, fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgMenu_Destroy(fgMenu* self);
FG_EXTERN char FG_FASTCALL fgMenu_Message(fgMenu* self, const FG_Msg* msg);
FG_EXTERN char FG_FASTCALL fgMenu_DoDropdown(fgMenu* self);

#ifdef  __cplusplus
}
#endif

#endif
