// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_BOX_H__
#define __FG_BOX_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGBOX_FLAGS
{
  FGBOX_IGNOREMARGINEDGEX = (FGSCROLLBAR_SHOWV << 1),
  FGBOX_IGNOREMARGINEDGEY = (FGBOX_IGNOREMARGINEDGEX << 1),
  FGBOX_TILEX = (FGBOX_IGNOREMARGINEDGEY << 1),
  FGBOX_TILEY = (FGBOX_TILEX << 1),
  FGBOX_DISTRIBUTEX = (FGBOX_TILEY << 1), // when combined with TILEX and TILEY, simply makes the tiles expand along the X direction
  FGBOX_DISTRIBUTEY = (FGBOX_DISTRIBUTEX << 1), // same as above but for the Y direction
  FGBOX_TILE = (FGBOX_TILEX | FGBOX_TILEY),
  FGBOX_DISTRIBUTE = (FGBOX_DISTRIBUTEX | FGBOX_DISTRIBUTEY),
  FGBOX_IGNOREMARGINEDGE = (FGBOX_IGNOREMARGINEDGEX | FGBOX_IGNOREMARGINEDGEY),
  FGBOX_LAYOUTMASK = (FGBOX_TILE | FGBOX_DISTRIBUTE | FGBOX_IGNOREMARGINEDGE),
};

struct _FG_BOX_ORDERED_ELEMENTS_ {
  char isordered; // If we detect that a BACKGROUND element was inserted in the middle of the foreground elements, we disable fgOrderedDraw until all children are removed.
  fgVectorElement ordered; // Used to implement fgOrderedDraw if TILEX or TILEY layouts are used.
  AbsVec fixedsize;
};
// A List is an arbitrary list of items with a number of different layout options that are selectable and/or draggable.
typedef struct _FG_BOX_ {
  fgScrollbar scroll;
  struct _FG_BOX_ORDERED_ELEMENTS_ order;
  void(*fndraw)(fgElement*, const AbsRect*, const fgDrawAuxData*);
#ifdef  __cplusplus
  inline operator fgElement*() { return &scroll.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgBox;

FG_EXTERN void FG_FASTCALL fgBox_Init(fgBox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units);
FG_EXTERN void FG_FASTCALL fgBox_Destroy(fgBox* self);
FG_EXTERN size_t FG_FASTCALL fgBox_Message(fgBox* self, const FG_Msg* msg);
FG_EXTERN void FG_FASTCALL fgBoxOrderedElement_Destroy(struct _FG_BOX_ORDERED_ELEMENTS_* self);
FG_EXTERN size_t FG_FASTCALL fgBoxOrderedElement_Message(struct _FG_BOX_ORDERED_ELEMENTS_* self, const FG_Msg* msg, fgElement* element, fgMessage callback);

#ifdef  __cplusplus
}
#endif

#endif