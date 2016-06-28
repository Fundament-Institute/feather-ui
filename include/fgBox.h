// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_BOX_H__
#define _FG_BOX_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGBOX_FLAGS
{
  FGBOX_TILEX = (FGSCROLLBAR_SHOWV << 1),
  FGBOX_TILEY = (FGBOX_TILEX << 1),
  FGBOX_DISTRIBUTEX = (FGBOX_TILEY << 1), // when combined with TILEX and TILEY, simply makes the tiles expand along the X direction
  FGBOX_DISTRIBUTEY = (FGBOX_DISTRIBUTEX << 1), // same as above but for the Y direction
  FGBOX_FIXEDX = (FGBOX_DISTRIBUTEY << 1), // This is a flag set for you by fgBox that tells it all elements are the same size, which allows for lookup optimizations.
  FGBOX_FIXEDY = (FGBOX_FIXEDX << 1),
  FGBOX_LAYOUTMASK = FGBOX_TILEX | FGBOX_TILEY | FGBOX_DISTRIBUTEX | FGBOX_DISTRIBUTEY,
  FGBOX_TILE = (FGBOX_TILEX | FGBOX_TILEY),
};

// A List is an arbitrary list of items with a number of different layout options that are selectable and/or draggable.
typedef struct _FG_BOX_ {
  fgScrollbar window;
  bool isordered; // If we detect that a BACKGROUND element was inserted in the middle of the foreground elements, we disable fgOrderedDraw until all children are removed.
  fgVectorElement ordered; // Used to implement fgOrderedDraw if TILEX or TILEY layouts are used.
#ifdef  __cplusplus
  inline operator fgElement*() { return &window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgBox;

FG_EXTERN void FG_FASTCALL fgBox_Init(fgBox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgBox_Destroy(fgBox* self);
FG_EXTERN size_t FG_FASTCALL fgBox_Message(fgBox* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif