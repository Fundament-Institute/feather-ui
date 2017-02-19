// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_EDITOR_H__
#define __FG_LAYOUT_EDITOR_H__

#include "fgAll.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG_LAYOUT_ACTION
{
  fgElement* self;
  FG_Msg msg;
  struct FG_LAYOUT_ACTION* next;
} fgLayoutAction;

typedef struct FG_LAYOUT_EDITOR
{
  fgWindow* window;
  fgMenu* menu;
  fgList* toolbar;
  fgTreeview* view;
  fgGrid* properties;
  fgElement* workspace;
  fgElement* selected;
  fgLayoutAction* undo;
  fgLayoutAction* redo;
} fgLayoutEditor;

FG_EXTERN void fgLayoutEditor_Init(fgLayoutEditor* self);
FG_EXTERN void fgLayoutEditor_Destroy(fgLayoutEditor* self);

#ifdef  __cplusplus
}
#endif

#endif