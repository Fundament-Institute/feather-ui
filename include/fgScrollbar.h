// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SCROLLBAR_H__
#define __FG_SCROLLBAR_H__

#include "fgControl.h"
#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGSCROLLBAR_FLAGS
{
  FGSCROLLBAR_HIDEH = (FGCONTROL_DISABLE << 1), // Never shows the Vertical or Horizontal scrollbar. Overrides SHOWV/SHOWH
  FGSCROLLBAR_HIDEV = (FGSCROLLBAR_HIDEH << 1),
  FGSCROLLBAR_SHOWH = (FGSCROLLBAR_HIDEV << 1), // Always show the Vertical or Horizontal scrollbar, even if not needed.
  FGSCROLLBAR_SHOWV = (FGSCROLLBAR_SHOWH << 1),
};

enum FGSCROLLBAR_ACTIONS
{
  FGSCROLLBAR_CHANGE = 1, // Triggered by the actual amount shifted
  FGSCROLLBAR_BAR,
  FGSCROLLBAR_BARINIT,
  FGSCROLLBAR_PAGE, // PageUp, PageDown, and click on the spaces between the scrollbars. 0 1 2 3 - left top right bottom
  FGSCROLLBAR_BUTTON, // Clicking the actual buttons. 0 1 2 3 - left top right bottom
  FGSCROLLBAR_BARCACHE, // resets the barcache
  FGSCROLLBAR_SCROLLTO, // Scrolls to include the given rect in the visible area relative to the parent. If this isn't possible, minimizes the amount of scrolling done while maximizing the visible area.
  FGSCROLLBAR_SCROLLTOABS, // Scrolls to include the given absolute rect in the visible area.
  FGSCROLLBAR_NUM,
};

struct _FG_SCROLLBAR_INNER {
  fgButton button;
  AbsVec lastmouse;
};

// A Scrollbar area acts as a clipping area for a single fgElement.
typedef struct _FG_SCROLLBAR {
  fgControl control;
  fgButton btn[4]; // 0 - left arrow, 1 - up arrow, 2 - right arrow, 3 - down arrow
  struct _FG_SCROLLBAR_INNER bar[2]; // 0 - horz slider, 1 - vert slider
  fgElement bg[3]; // 0 - horizontal background, 1 - vertical background, 2 - corner
  AbsRect realpadding; // We have to intercept and store padding amounts here because we hijack the padding to perform scrolling
  AbsVec barcache; // Stores scrollbar width/height
  AbsVec realsize; // Stores the total size of the children calculated from the layout.
  AbsVec lastpadding;
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgScrollbar;

FG_EXTERN void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self);
FG_EXTERN size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif