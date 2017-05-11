// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __LAYOUT_TAB_H__
#define __LAYOUT_TAB_H__

#include "EditorBase.h"
#include "bss-util/Hash.h"

class LayoutTab
{
public:
  LayoutTab();
  ~LayoutTab();
  void Init(EditorBase* base);
  void OpenLayout(fgLayout* layout);
  void AddProp(const char* name, FG_UINT id, const char* type = "textbox");
  void Clear();
  void Link(fgElement* e, fgClassLayout* layout);
  inline void ClearLinks() { _elementmap.Clear(); _layoutmap.Clear(); }

  void MenuInsert(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuAdd(struct _FG_ELEMENT*, const FG_Msg*);
  void MenuContext(struct _FG_ELEMENT*, const FG_Msg*);

protected:
  void _openSublayout(fgElement* root, fgLayout* parent);
  void _openLayout(fgElement* root, const fgVectorClassLayout& layout);
  void _doInsert(const char* type, bool insert);
  void _treeItemOnFocus(struct _FG_ELEMENT*, const FG_Msg*);
  void _textboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg);
  void _editboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg);
  void _treeviewOnMouseDown(struct _FG_ELEMENT* e, const FG_Msg* msg);

  // Message handler for the edit box
  static size_t _treeviewMessage(fgTreeview* self, const FG_Msg* msg);

  fgTextbox _editbox;
  EditorBase* _base;
  fgTreeview* _layoutview;
  fgGrid* _layoutprops;
  fgElement* _contextmenu;
  fgElement* _contextmenulayout;
  fgElement* _selected; // Current selected item in the tree, if applicable
  bss::Hash<fgElement*, fgClassLayout*> _elementmap; // Map of elements to their corresponding class layout
  bss::Hash<fgClassLayout*, fgElement*> _layoutmap; // reverse hash of the above
  std::function<void(fgElement*, const char*)> _propafter;
};

#endif
