// Copyright ©2012 Black Sphere Studios
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
  FGSCROLLBAR_HIDEX = (1 << 8),
  FGSCROLLBAR_HIDEY = (1 << 9),
  FGSCROLLBAR_PUSHX = (1 << 10),
  FGSCROLLBAR_PUSHY = (1 << 11),
};

// A Scrollbar area acts as a clipping area for a single fgChild.
typedef struct {
  fgWindow window;
  fgChild target; // This is the child that we should be using scrollbars to navigate, if necessary
  fgButton btn[6]; // 0 - up arrow, 1 - down arrow, 2 - vertical slider, 3 - right arrow, 4 - left arrow, 6 - horz slider
  fgChild bg[2]; // 0 - vertical background, 1 - horizontal background
} fgScrollbar;

FG_EXTERN void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif