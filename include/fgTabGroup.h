// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TABGROUP_H__
#define __FG_TABGROUP_H__

#include "fgContainer.h"
#include "fgButton.h"

#ifdef  __cplusplus
extern "C" {
#endif

// A Tab group is a collection of tab pages that can be switched between by clicking on "tabs".
typedef struct {
  fgContainer window;
  fgChild* skin[1]; // 0 - background
} fgTabGroup;

FG_EXTERN fgWindow* FG_FASTCALL fgTabGroup_Create(fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN void FG_FASTCALL fgTabGroup_Init(fgTabGroup* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags);
FG_EXTERN char FG_FASTCALL fgTabGroup_Message(fgTabGroup* self, const FG_Msg* msg);

#ifdef  __cplusplus
}
#endif

#endif