// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"
#include "bss-util/bss_util.h"

using namespace bss;

LayoutTab::LayoutTab()
{
  bssFill(*this, 0);
}
LayoutTab::~LayoutTab()
{
}
void LayoutTab::Init(EditorBase* base)
{
  _base = base;
  _layoutview = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  _layoutprops = reinterpret_cast<fgGrid*>(fgGetID("Editor$layoutprops"));

  if(_layoutprops)
  {
    AddProp("(Type)", 0, "text");
    AddProp("ID", EditorBase::PROP_ID);
    AddProp("Name", EditorBase::PROP_NAME);
    AddProp("Skin", EditorBase::PROP_SKIN, "combobox");
    AddProp("Transform", 0, "text");
    AddProp("  Area", EditorBase::PROP_AREA);
    AddProp("  Rotation", EditorBase::PROP_ROTATION);
    AddProp("  Center", EditorBase::PROP_CENTER);
    AddProp("Margin", EditorBase::PROP_MARGIN);
    AddProp("Padding", EditorBase::PROP_PADDING);
    AddProp("Min Dim", EditorBase::PROP_MINDIM);
    AddProp("Max Dim", EditorBase::PROP_MAXDIM);
    AddProp("Scaling", EditorBase::PROP_SCALING);
    AddProp("Order", EditorBase::PROP_ORDER);
    AddProp("Style", EditorBase::PROP_STYLE);
    AddProp("Flags", EditorBase::PROP_FLAGS, "text");
    AddProp("Alpha", EditorBase::PROP_ALPHA, "slider");
    (*_layoutprops)->userdata = this;
  }
}

void LayoutTab::AddProp(const char* name, FG_UINT id, const char* type)
{
  fgFlag flags = FGELEMENT_EXPANDY;
  fgMessage f = nullptr;
  if(!STRICMP(type, "textbox"))
  {
    flags |= FGTEXTBOX_ACTION | FGTEXTBOX_SINGLELINE;
    f = (fgMessage)_textboxMessage;
  }
  _base->AddProp(*_layoutprops, name, type, id, f, flags);
}

size_t LayoutTab::_textboxMessage(fgTextbox* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;
  switch(msg->type)
  {
  case FG_ACTION:
    if(!msg->subtype)
    {
      auto tab = (LayoutTab*)(*self)->parent->parent->userdata;
      if(tab->_selected && tab->_selected->userdata)
      {
        auto p = (fgClassLayout*)tab->_selected->userdata;
        tab->_base->ParseStyleMsg(p->layout.style, tab->_layoutmap[p], 0, p, EditorBase::PROPERTIES((*self)->userid), (*self)->GetText());
        tab->_base->LoadProps(*tab->_layoutprops, *p);
      }
    }
    break;
  }

  return fgTextbox_Message(self, msg);
}


void LayoutTab::_openLayout(fgElement* root, const fgVectorClassLayout& layout)
{
  for(size_t i = 0; i < layout.l; ++i)
  {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = layout.p + i;
    fgElement_AddListener(item, FG_GOTFOCUS, &_treeItemOnFocus);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(layout.p[i].layout.type);
    _openLayout(item, layout.p[i].children);
  }
}
void LayoutTab::OpenLayout(fgLayout* layout)
{
  if(_layoutview)
  {
    fgElement_Clear(*_layoutview);

    fgCreate("Text", *_layoutview, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText("root");
    _openLayout(*_layoutview, layout->layout);
  }
}
void LayoutTab::OnFocus(fgElement* e)
{
  auto p = (fgClassLayout*)e->userdata;
  if(p)
  {
    _selected = e;
    _base->LoadProps(*_layoutprops, *p);
  }
}
void LayoutTab::_treeItemOnFocus(struct _FG_ELEMENT* e, const FG_Msg*)
{
  fgLayoutEditor::Instance->Layout().OnFocus(e);
}

void LayoutTab::Clear()
{
  if(_layoutview)
    fgElement_Clear(*_layoutview);
  if(_layoutprops)
    _base->ClearProps(*_layoutprops);
  ClearLinks();
}
void LayoutTab::Link(fgElement* e, fgClassLayout* layout)
{
  _elementmap.Insert(e, layout);
  _layoutmap.Insert(layout, e);
}
