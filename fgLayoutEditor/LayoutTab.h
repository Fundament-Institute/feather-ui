// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __LAYOUT_TAB_H__
#define __LAYOUT_TAB_H__

#include "EditorBase.h"

class LayoutTab
{
public:
  LayoutTab();
  ~LayoutTab();
  void Init(EditorBase* base);
  void OpenLayout(fgLayout* layout);
  void AddProp(const char* name, FG_UINT id, const char* type = "text");
  void OnFocus(fgElement* e);
  void Clear();

protected:
  void _openLayout(fgElement* root, const fgVectorClassLayout& layout);

  // Action listener on the property grid
  static void _layoutPropsAction(fgElement* e, const FG_Msg* m);
  // Message handler for text properties that brings up the editbox
  static size_t _propertyMessage(fgText* self, const FG_Msg* msg);
  // Message handler for the edit box
  static size_t _editboxMessage(fgTextbox* self, const FG_Msg* msg);
  static void _treeItemOnFocus(struct _FG_ELEMENT*, const FG_Msg*);

  EditorBase* _base;
  fgTreeview* _layoutview;
  fgGrid* _layoutprops;
  fgTreeItem* _selected; // Current selected item in the tree, if applicable
  fgTextbox _editbox;
  AbsVec _lastscroll;
};

#endif
