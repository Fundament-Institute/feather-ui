// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "SkinTab.h"
#include "bss-util/bss_util.h"

using namespace bss;

SkinTab::SkinTab()
{
  bssFill(*this, 0);
}
SkinTab::~SkinTab()
{
}
void SkinTab::Init(EditorBase* base)
{
  _base = base;
  _skinview = reinterpret_cast<fgTreeview*>(fgGetID("Editor$skin"));
  fgElement_AddDelegateListener<SkinTab, &SkinTab::_treeviewOnMouseDown>(*_skinview, FG_MOUSEDOWN, this);

  _skinprops = reinterpret_cast<fgGrid*>(fgGetID("Editor$skinprops"));
  _contextmenu = fgGetID("Editor$skincontext");
  fgElement_AddDelegateListener<SkinTab, &SkinTab::MenuContext>(_contextmenu, FG_ACTION, this);

  fgTransform tfeditbox = { { 0, 0, 0, 0, 50, 0, 0, 0 }, 0,{ 0,0,0,0 } };
  fgTextbox_Init(&_editbox, 0, 0, "Editor$skineditbox", FGELEMENT_EXPANDY | FGTEXTBOX_SINGLELINE | FGTEXTBOX_ACTION, &tfeditbox, 0);
  fgElement_AddDelegateListener<SkinTab, &SkinTab::_editboxOnAction>(_editbox, FG_ACTION, this);
  fgElement_AddDelegateListener<SkinTab, &SkinTab::_editboxOnAction>(_editbox, FG_LOSTFOCUS, this);

  _propafter = [this](fgElement* e, const char* type) {
    if(!STRICMP(type, "textbox"))
    {
      fgElement_AddDelegateListener<SkinTab, &SkinTab::_textboxOnAction>(e, FG_ACTION, this);
      fgElement_AddDelegateListener<SkinTab, &SkinTab::_textboxOnAction>(e, FG_LOSTFOCUS, this);
    }
  };

  if(_skinprops)
  {
    AddProp("(Type)", 0, "text");
    AddProp("ID", EditorBase::PROP_ID);
    AddProp("Name", EditorBase::PROP_NAME);
    AddProp("Skin", EditorBase::PROP_SKIN, "text");
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
    AddProp("Flags", EditorBase::PROP_FLAGS, "box");
    AddProp("Alpha", EditorBase::PROP_ALPHA);
  }
}
void SkinTab::_openSkinBase(fgElement* root, fgSkinBase& skin)
{
  std::function<void(fgSkin*, const char*)> fn = [this, root](fgSkin* s, const char* name) {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = s;
    item->userid = EditorBase::TYPE_SKIN;
    fgElement_AddDelegateListener<SkinTab, &SkinTab::_treeItemOnFocus>(item, FG_GOTFOCUS, this);
    item->SetContextMenu(_contextmenu);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(name);
    _openSkinBase(item, s->base);
    _openSkinTree(item, s->tree);
  };

  fgSkinBase_IterateSkins(&skin, &fn, &Delegate<void, fgSkin*, const char*>::StubLambda);
}
void SkinTab::WriteStyleMap(Str& s, FG_UINT map)
{
  while(map)
  {
    FG_UINT index = (1 << bssLog2(map));
    if(const char* name = fgStyle_GetMapIndex(index))
    {
      s += name;
      s += "+";
    }
    map ^= index;
  }
  if(s.size() > 1) // chop off trailing +
    s.resize(s.size() - 1);
}
void SkinTab::_openSkinTree(fgElement* root, fgSkinTree& skin)
{
  for(size_t i = 0; i < skin.styles.l; ++i)
  {
    Str s;
    WriteStyleMap(s, skin.styles.p[i].map);
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = skin.styles.p + i;
    item->userid = EditorBase::TYPE_STYLE;
    fgElement_AddDelegateListener<SkinTab, &SkinTab::_treeItemOnFocus>(item, FG_GOTFOCUS, this);
    item->SetContextMenu(_contextmenu);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(s);
  }
  for(size_t i = 0; i < skin.children.l; ++i)
  {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = skin.children.p + i;
    item->userid = EditorBase::TYPE_SKINLAYOUT;
    fgElement_AddDelegateListener<SkinTab, &SkinTab::_treeItemOnFocus>(item, FG_GOTFOCUS, this);
    item->SetContextMenu(_contextmenu);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(skin.children.p[i].element.type);
    _openSkinTree(item, skin.children.p[i].tree);
  }
}
void SkinTab::OpenSkin(fgSkin* skin)
{
  if(_skinview)
  {
    fgElement_Clear(*_skinview);

    fgElement* root = fgCreate("Text", *_skinview, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    root->SetText("root");
    (*_skinview)->userdata = skin;
    (*_skinview)->userid = EditorBase::TYPE_SKIN;
    fgElement_AddDelegateListener<SkinTab, &SkinTab::_treeviewOnMouseDown>(root, FG_MOUSEDOWN, this);
    _openSkinBase(*_skinview, skin->base);
    _openSkinTree(*_skinview, skin->tree);
  }
}
void SkinTab::Clear()
{
  if(_skinview)
    fgElement_Clear(*_skinview);
  if(_skinprops)
    _base->ClearProps(*_skinprops);
}
void SkinTab::AddProp(const char* name, FG_UINT id, const char* type)
{
  fgFlag flags = FGELEMENT_EXPANDY;
  fgMessage f = nullptr;
  if(!STRICMP(type, "textbox"))
    flags |= FGTEXTBOX_ACTION | FGTEXTBOX_SINGLELINE;
  if(!STRICMP(type, "box"))
    flags |= FGBOX_TILEY;
  _propafter(_base->AddProp(*_skinprops, name, type, id, f, flags), type);
}

void SkinTab::InsertElement(fgElement* e, const char* type, const char* name, bool insert, fgTransform* tf)
{
  auto playout = (fgSkinLayout*)e->userdata;
  auto pskin = (fgSkin*)e->userdata;

  switch(e->userid)
  {
  case EditorBase::TYPE_SKINLAYOUT:
    if(!STRICMP(type, "skin"))
      return; // Can't insert skin to skinlayout
    if(!insert)
    {
      if(!STRICMP(type, "style"))
        playout->tree.AddStyle(name);
      else
        playout->tree.AddChild(type, !tf ? FGELEMENT_EXPAND : 0, !tf ? &fgTransform_EMPTY : tf, 0, 0);
    }
    fgElement_ClearType(e, "treeitem");
    _openSkinTree(e, playout->tree);
    break;
  case EditorBase::TYPE_SKIN:
    if(!insert)
    {
      if(!STRICMP(type, "skin"))
        pskin->base.AddSkin(name);
      else if(!STRICMP(type, "style"))
        pskin->tree.AddStyle(name);
      else
        pskin->tree.AddChild(type, !tf ? FGELEMENT_EXPAND : 0, !tf ? &fgTransform_EMPTY : tf, 0, 0);
    }

    fgElement_ClearType(e, "treeitem");
    _openSkinBase(e, pskin->base);
    _openSkinTree(e, pskin->tree);
    break;
  }
}

void SkinTab::RemoveElement(fgElement* e)
{
  if(e && e->userdata && e->parent && e->parent->userdata)
  {
    fgElement* parent = e->parent;
    auto playout = (fgSkinLayout*)parent->userdata;
    auto pskin = (fgSkin*)parent->userdata;
    switch(parent->userid)
    {
    case EditorBase::TYPE_SKINLAYOUT:
      switch(e->userid)
      {
      case EditorBase::TYPE_SKINLAYOUT:
        playout->tree.RemoveChild(reinterpret_cast<fgSkinLayout*>(e->userdata) - playout->tree.children.p);
        break;
      case EditorBase::TYPE_STYLE:
        playout->tree.RemoveStyle(reinterpret_cast<fgSkinTree::fgStylePair*>(e->userdata)->map);
        break;
      }
      fgElement_ClearType(parent, "treeitem");
      _openSkinTree(parent, playout->tree);
      break;
    case EditorBase::TYPE_SKIN:
      switch(e->userid)
      {
      case EditorBase::TYPE_SKIN:
        pskin->base.RemoveSkin(reinterpret_cast<fgSkin*>(e->userdata)->base.name);
        break;
      case EditorBase::TYPE_SKINLAYOUT:
        pskin->tree.RemoveChild(reinterpret_cast<fgSkinLayout*>(e->userdata) - pskin->tree.children.p);
        break;
      case EditorBase::TYPE_STYLE:
        pskin->tree.RemoveStyle(reinterpret_cast<fgSkinTree::fgStylePair*>(e->userdata)->map);
        break;
      }
      _base->ReapplySkin(pskin);
      fgElement_ClearType(parent, "treeitem");
      _openSkinBase(parent, pskin->base);
      _openSkinTree(parent, pskin->tree);
      break;
    }
  }
}

void SkinTab::MenuContext(struct _FG_ELEMENT*, const FG_Msg* m)
{
  if(m->e)
  {
    switch(m->e->userid)
    {
    case 1:
      InsertElement(_selected, "skin", "newskin", false, 0);
      break;
    case 2:
      InsertElement(_selected, "style", "neutral", false, 0);
      break;
    case 3:
      InsertElement(_selected, "resource", 0, false, 0);
      break;
    case 4:
      InsertElement(_selected, "text", 0, false, 0);
      break;
    case 5:
      InsertElement(_selected, "curve", 0, false, 0);
      break;
    case 7:
      RemoveElement(_selected);
      break;
    }
  }
}

void SkinTab::_textboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg)
{
  if(!msg->subtype && _selected && _selected->userdata)
  {
    fgSkin* skin;
    fgSkinTree::fgStylePair* style;
    fgSkinLayout* layout;

    switch(_selected->userid)
    {
    case EditorBase::TYPE_SKINLAYOUT:
      layout = (fgSkinLayout*)_selected->userdata;
      _base->ParseStyleMsg(layout->element.style, layout->instance, &layout->element, 0, &layout->element.skin, 0, EditorBase::PROPERTIES(e->userid), e->GetText());
      _base->SetProps(*_skinprops, 0, &layout->element, layout->element.skin, 0, layout->element.style);
      break;
    case EditorBase::TYPE_SKIN:
      skin = (fgSkin*)_selected->userdata;
      _base->ParseStyleMsg(skin->base.style, 0, 0, 0, 0, &skin->base.name, EditorBase::PROPERTIES(e->userid), e->GetText());
      _base->ReapplySkin(skin);
      _base->SetProps(*_skinprops, 0, 0, 0, skin->base.name, skin->base.style);
      break;
    case EditorBase::TYPE_STYLE:
      style = (fgSkinTree::fgStylePair*)_selected->userdata;
      _base->ParseStyleMsg(style->style, 0, 0, 0, 0, 0, EditorBase::PROPERTIES(e->userid), e->GetText());
      _base->SetProps(*_skinprops, 0, 0, 0, 0, style->style);
      break;
    }
  }
}

void SkinTab::_editboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg)
{

}
void SkinTab::_treeviewOnMouseDown(struct _FG_ELEMENT* e, const FG_Msg* msg)
{
  _selected = 0;
  (*_skinprops)->GotFocus();
}
void SkinTab::_treeItemOnFocus(struct _FG_ELEMENT* e, const FG_Msg*)
{
  if(e->userdata)
  {
    fgSkin* skin;
    fgSkinTree::fgStylePair* style;
    fgSkinLayout* layout;

    switch(e->userid)
    {
    case EditorBase::TYPE_SKINLAYOUT:
      layout = (fgSkinLayout*)e->userdata;
      _base->LoadProps(*_skinprops, layout->element.type, 0, &layout->element, layout->element.skin, 0, layout->element.style, _propafter);
      break;
    case EditorBase::TYPE_SKIN:
      skin = (fgSkin*)e->userdata;
      _base->LoadProps(*_skinprops, "Skin", 0, 0, 0, skin->base.name, skin->base.style, _propafter);
      break;
    case EditorBase::TYPE_STYLE:
      style = (fgSkinTree::fgStylePair*)e->userdata;
      if(e->parent && e->parent->userdata)
      {
        switch(e->parent->userid)
        {
        case EditorBase::TYPE_SKINLAYOUT:
          _base->LoadProps(*_skinprops, ((fgSkinLayout*)e->parent->userdata)->element.type, 0, 0, 0, 0, style->style, _propafter);
          break;
        case EditorBase::TYPE_SKIN:
          _base->LoadProps(*_skinprops, "Skin", 0, 0, 0, 0, style->style, _propafter);
          break;
        default:
          _base->LoadProps(*_skinprops, 0, 0, 0, 0, 0, style->style, _propafter);
          break;
        }
      }
      break;
    }
    _selected = e;
  }
}
