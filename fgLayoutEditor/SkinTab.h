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
  void OpenSkin(fgSkin* skin);
  void Clear();
  void AddProp(const char* name, FG_UINT id, const char* type = "textbox");
  void InsertElement(fgElement* e, const char* type, const char* name, bool insert, fgTransform* tf);
  void RemoveElement(fgElement* e);
  inline fgElement* GetSelected() { return _selected; }

  void MenuContext(struct _FG_ELEMENT*, const FG_Msg*);

protected:
  void _textboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg);
  void _editboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg);
  void _treeviewOnMouseDown(struct _FG_ELEMENT* e, const FG_Msg* msg);
  void _treeItemOnFocus(struct _FG_ELEMENT*, const FG_Msg*);
  void _openSkinBase(fgElement* root, fgSkinBase& skin);
  void _openSkinTree(fgElement* root, fgSkinTree& skin);

  static void WriteStyleMap(bss::Str& s, FG_UINT map);

  fgTextbox _editbox;
  EditorBase* _base;
  fgTreeview* _skinview;
  fgGrid* _skinprops;
  fgElement* _contextmenu;
  fgElement* _selected; // Current selected item in the tree, if applicable
  std::function<void(fgElement*, const char*)> _propafter;
};

#endif
