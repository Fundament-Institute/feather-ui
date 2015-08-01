// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LIST_H__
#define __FG_LIST_H__

#include "fgGrid.h"
#include "fgScrollbar.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A List is a list of items that can be sorted into any number of columns and optionally have column headers.
typedef struct {
  fgScrollbar window;
  fgGrid grid;
  fgStatic* selector;
  fgStatic* highlighter;
} fgList;

FG_EXTERN fgWindow* FG_FASTCALL fgList_Create(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif