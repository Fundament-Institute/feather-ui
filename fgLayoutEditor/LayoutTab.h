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
  void OnFocus(fgElement* e);
  void Clear();
  void Link(fgElement* e, fgClassLayout* layout);
  inline void ClearLinks() { _elementmap.Clear(); _layoutmap.Clear(); }

protected:
  void _openLayout(fgElement* root, const fgVectorClassLayout& layout);

  // Message handler for the edit box
  static size_t _textboxMessage(fgTextbox* self, const FG_Msg* msg);
  static void _treeItemOnFocus(struct _FG_ELEMENT*, const FG_Msg*);

  EditorBase* _base;
  fgTreeview* _layoutview;
  fgGrid* _layoutprops;
  fgElement* _selected; // Current selected item in the tree, if applicable
  bss::Hash<fgElement*, fgClassLayout*> _elementmap; // Map of elements to their corresponding class layout
  bss::Hash<fgClassLayout*, fgElement*> _layoutmap; // reverse hash of the above
  AbsVec _lastscroll;
};

#endif
