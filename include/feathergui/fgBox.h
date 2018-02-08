// Copyright ©2018 Black Sphere Studios
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
    FGBOX_REVERSE = (FGBOX_TILEY << 1),
    FGBOX_GROWX = 0,
    FGBOX_GROWY = (FGBOX_REVERSE << 1),
    FGBOX_DISTRIBUTE = (FGBOX_GROWY << 1),
    FGBOX_TILE = (FGBOX_TILEX | FGBOX_TILEY),
    FGBOX_IGNOREMARGINEDGE = (FGBOX_IGNOREMARGINEDGEX | FGBOX_IGNOREMARGINEDGEY),
    FGBOX_LAYOUTMASK = (FGBOX_TILE | FGBOX_GROWX | FGBOX_GROWY | FGBOX_REVERSE | FGBOX_DISTRIBUTE | FGBOX_IGNOREMARGINEDGE),
  };

  struct _FG_BOX_ORDERED_ELEMENTS_ {
    char isordered; // If we detect that a BACKGROUND element was inserted in the middle of the foreground elements, we disable fgOrderedDraw until all children are removed.
    fgVectorElement ordered; // Used to implement fgOrderedDraw if TILEX or TILEY layouts are used.
  };

  struct _FG_SKIN;

  // A List is an arbitrary list of items with a number of different layout options that are selectable and/or draggable.
  typedef struct _FG_BOX_ {
    fgScrollbar scroll;
    struct _FG_BOX_ORDERED_ELEMENTS_ order;
    void(*fndraw)(fgElement*, const AbsRect*, const fgDrawAuxData*, fgElement*);
    fgColor dividercolor;
    struct _FG_SKIN* dividerskin;
    fgVectorElement selected; // While fgBox does not handle selecting elements, it must have access to them if they exist
    AbsVec spacing;
    AbsVec fixedsize;

#ifdef  __cplusplus
    inline operator fgElement*() { return &scroll.control.element; }
    inline fgElement* operator->() { return operator fgElement*(); }
#endif
  } fgBox;

  FG_EXTERN void fgBox_Init(fgBox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units);
  FG_EXTERN void fgBox_Destroy(fgBox* self);
  FG_EXTERN size_t fgBox_Message(fgBox* self, const FG_Msg* msg);
  FG_EXTERN void fgBoxOrderedElement_Destroy(struct _FG_BOX_ORDERED_ELEMENTS_* self);
  FG_EXTERN size_t fgBoxOrderedElement_Message(struct _FG_BOX_ORDERED_ELEMENTS_* self, const FG_Msg* msg, fgElement* element, fgMessage callback, const AbsVec* spacing);
  FG_EXTERN fgElement* fgBoxOrderedElement_Get(struct _FG_BOX_ORDERED_ELEMENTS_* self, const AbsRect* target, const AbsRect* area, fgFlag flags);
  FG_EXTERN void fgBox_DeselectAll(fgBox* self);
  FG_EXTERN void fgBox_SelectTarget(fgBox* self, fgElement* target);
  FG_EXTERN void fgBoxRenderDividers(fgElement* self, fgColor color, const struct _FG_SKIN* skin, const AbsRect* area, const AbsRect* drawarea, const fgDrawAuxData* aux, fgElement* begin);

#ifdef  __cplusplus
}
#endif

#endif