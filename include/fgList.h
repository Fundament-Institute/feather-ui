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
  FGLIST_SELECT = (FGBOX_DISTRIBUTEY << 1),
  FGLIST_MULTISELECT = (FGLIST_SELECT << 1) | FGLIST_SELECT,
  FGLIST_DRAGGABLE = (FGLIST_MULTISELECT << 1),
};

// A List is an arbitrary list of items with a number of different layout options that are selectable and/or draggable.
typedef struct {
  fgBox box;
  fgColor select; // color index 0
  fgColor hover; // color index 1
  fgColor drag; // color index 2
  fgVectorElement selected; // vector of all selected elements
  fgMouseState mouse;
#ifdef  __cplusplus
  inline operator fgElement*() { return &box.window.control.element; }
  inline fgElement* operator->() { return operator fgElement*(); }
#endif
} fgList;

FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN void FG_FASTCALL fgList_Destroy(fgList* self);
FG_EXTERN size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg);

FG_EXTERN void FG_FASTCALL fgListItem_Init(fgControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform);
FG_EXTERN size_t FG_FASTCALL fgListItem_Message(fgControl* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif