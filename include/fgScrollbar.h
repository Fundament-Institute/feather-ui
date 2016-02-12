// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SCROLLBAR_H__
#define __FG_SCROLLBAR_H__

#include "fgWindow.h"
#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGSCROLLBAR_FLAGS
{
  FGSCROLLBAR_VERT = (1 << 6),
  FGSCROLLBAR_HORZ = (1 << 7),
  FGSCROLLBAR_SHOWX = (1 << 8),
  FGSCROLLBAR_SHOWY = (1 << 9),
};

enum FGSCROLLBAR_ACTIONS
{
  FGSCROLLBAR_CHANGE = 0, // Triggered by the actual amount shifted
  FGSCROLLBAR_VDELTA, // vertical mouse wheel
  FGSCROLLBAR_HDELTA, // horizontal mouse wheel
  FGSCROLLBAR_PAGE, // PageUp, PageDown, and click on the spaces between the scrollbars. 0 1 2 3 - left top right bottom
  FGSCROLLBAR_BUTTON, // Clicking the actual buttons. 0 1 2 3 - left top right bottom
};

// A Scrollbar area acts as a clipping area for a single fgChild.
typedef struct {
  fgWindow window;
  fgButton btn[6]; // 0 - up arrow, 1 - down arrow, 2 - vertical slider, 3 - right arrow, 4 - left arrow, 6 - horz slider
  fgChild bg[2]; // 0 - vertical background, 1 - horizontal background
  CVec maxdim;
  AbsRect realpadding; // We have to intercept and store padding amounts here because we hijack the padding to perform scrolling
  AbsVec barcache; // Stores scrollbar width/height
  AbsVec realsize; // Stores the total size of the children calculated from the layout.
} fgScrollbar;

FG_EXTERN void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self);
FG_EXTERN size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif