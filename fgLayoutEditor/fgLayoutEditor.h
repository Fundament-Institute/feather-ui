// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_EDITOR_H__
#define __FG_LAYOUT_EDITOR_H__

#include "../include/fgImplementation.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG_LAYOUT_ACTION
{
  fgChild* self;
  FG_Msg msg;
  struct FG_LAYOUT_ACTION* next;
} fgLayoutAction;

typedef struct FG_LAYOUT_EDITOR
{
  fgImplementation* dll;
  fgRoot* root;
  fgTopWindow* window;
  fgMenu* menu;
  fgList* toolbar;
  //fgTreeView* view;
  //fgGrid* properties;
  fgChild workspace;
  fgChild* selected;
  fgLayoutAction* undo;
  fgLayoutAction* redo;
} fgLayoutEditor;

extern fgLayoutEditor* LoadLayoutEditor(fgImplementation* dll);
extern void CloseLayoutEditor(fgLayoutEditor* editor);

#ifdef  __cplusplus
}
#endif

#endif