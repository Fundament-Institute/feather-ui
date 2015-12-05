// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LIST_H__
#define __FG_LIST_H__

#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum FGLIST_FLAGS
{
  FGLIST_TILEX = (1 << 12),
  FGLIST_TILEY = (1 << 13),
  FGLIST_DISTRIBUTEX = (1 << 14), // when combined with TILEX and TILEY, simply makes the tiles expand along the X direction
  FGLIST_DISTRIBUTEY = (1 << 15), // same as above but for the Y direction
};

// A List is a list of items that can be sorted into any number of columns and optionally have column headers.
typedef struct {
  fgScrollbar window;
  fgChild selector;
  fgChild highlight;
  fgChild* hover;
  fgChild* selected;
} fgList;

FG_EXTERN fgChild* FG_FASTCALL fgList_Create(fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgList_Destroy(fgList* self);
FG_EXTERN char FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif