// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_EDITOR_H__
#define __FG_LAYOUT_EDITOR_H__

#include "fgAll.h"

typedef struct FG_LAYOUT_ACTION
{
  fgElement* self;
  FG_Msg msg;
  struct FG_LAYOUT_ACTION* next;
} fgLayoutAction;

class fgLayoutEditor
{
public:
  explicit fgLayoutEditor(fgLayout* layout);
  ~fgLayoutEditor();

  static void MenuNew(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuOpen(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuSave(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuSaveAs(struct _FG_ELEMENT*, const FG_Msg*);
  static void MenuExit(struct _FG_ELEMENT*, const FG_Msg*);

  static fgLayoutEditor* Instance;

protected:
  static size_t _inject(fgRoot* self, const FG_Msg* msg);

  fgWindow* _mainwindow;
  fgTreeview* _explorer;
  fgGrid* _properties;
  fgTreeview* _skinexplorer;
  fgGrid* _skinprops;
  fgWorkspace* _workspace;

  fgLayout* _layout;
  fgElement* _selected;
  fgLayoutAction* _undo;
  fgLayoutAction* _redo;
};

#endif