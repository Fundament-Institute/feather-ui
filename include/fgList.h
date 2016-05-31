// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_LIST_H__
#define _FG_LIST_H__

#include "fgBox.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGLIST_FLAGS
{
  FGLIST_SINGLESELECT = (FGBOX_FIXEDY << 1),
  FGLIST_MULTISELECT = (FGLIST_SINGLESELECT << 1),
  FGLIST_DRAGGABLE = (FGLIST_MULTISELECT << 1),
};

// A List is an arbitrary list of items with a number of different layout options that are selectable and/or draggable.
typedef struct {
  fgBox box;
  fgElement selector;
  fgElement hover;
  fgElement* selected; // points to current selected item
  fgMouseState mouse;
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgList;

FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgList_Destroy(fgList* self);
FG_EXTERN size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif