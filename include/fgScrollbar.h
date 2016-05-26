// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_SCROLLBAR_H__
#define _FG_SCROLLBAR_H__

#include "fgControl.h"
#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGSCROLLBAR_FLAGS
{
  FGSCROLLBAR_HIDEH = (1 << 8), // Never shows the Vertical or Horizontal scrollbar. Overrides SHOWV/SHOWH
  FGSCROLLBAR_HIDEV = (1 << 9),
  FGSCROLLBAR_SHOWH = (1 << 10), // Always show the Vertical or Horizontal scrollbar, even if not needed.
  FGSCROLLBAR_SHOWV = (1 << 11),
};

enum FGSCROLLBAR_ACTIONS
{
  FGSCROLLBAR_CHANGE = 0, // Triggered by the actual amount shifted
  FGSCROLLBAR_PAGE, // PageUp, PageDown, and click on the spaces between the scrollbars. 0 1 2 3 - left top right bottom
  FGSCROLLBAR_BUTTON, // Clicking the actual buttons. 0 1 2 3 - left top right bottom
};

// A Scrollbar area acts as a clipping area for a single fgElement.
typedef struct {
  fgControl control;
  fgButton btn[6]; // 0 - left arrow, 1 - right arrow, 2 - horz slider, 3 - up arrow, 4 - down arrow, 5 - vert slider
  fgElement bg[2]; // 0 - horizontal background, 1 - vertical background
  AbsVec maxdim;
  AbsRect realpadding; // We have to intercept and store padding amounts here because we hijack the padding to perform scrolling
  AbsVec barcache; // Stores scrollbar width/height
  AbsVec realsize; // Stores the total size of the children calculated from the layout.
#ifdef  __cplusplus
  inline operator fgElement*() { return &control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgScrollbar;

FG_EXTERN void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self);
FG_EXTERN size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif