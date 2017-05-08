// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __SKIN_TAB_H__
#define __SKIN_TAB_H__

#include "EditorBase.h"

class SkinTab
{
public:
  SkinTab();
  ~SkinTab();
  void Init(EditorBase* base);
  void Clear();

protected:
  EditorBase* _base;
  fgTreeview* _skinview;
  fgGrid* _skinprops;
  fgTreeItem* _selected; // Current selected item in the tree, if applicable
  fgTextbox _editbox;
  AbsVec _lastscroll;
};

#endif
