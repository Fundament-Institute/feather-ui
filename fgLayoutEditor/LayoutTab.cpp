// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"
#include "bss-util/bss_util.h"

using namespace bss;

LayoutTab::LayoutTab()
{
  bssFill(*this, 0);
  fgRegisterDelegate<LayoutTab, &LayoutTab::MenuInsert>("LayoutMenuInsert", this);
  fgRegisterDelegate<LayoutTab, &LayoutTab::MenuAdd>("LayoutMenuAdd", this);
  fgRegisterDelegate<LayoutTab, &LayoutTab::MenuContext>("LayoutMenuContext", this);
}
LayoutTab::~LayoutTab()
{
}
fgLayout* LayoutTab::FindParentLayout(fgElement* treeitem)
{
  if(treeitem->userid == 1)
    return (fgLayout*)treeitem->userdata;
  if(!treeitem->parent)
    return 0;
  return FindParentLayout(treeitem->parent);
}
void LayoutTab::Init(EditorBase* base)
{
  _base = base;
  _layoutview = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_treeviewOnMouseDown>(*_layoutview, FG_MOUSEDOWN, this);

  _layoutprops = reinterpret_cast<fgGrid*>(fgGetID("Editor$layoutprops"));
  _contextmenu = fgGetID("Editor$layoutcontext");
  _contextmenulayout = fgGetID("Editor$sublayoutcontext");
  fgElement_AddDelegateListener<LayoutTab, &LayoutTab::MenuContext>(_contextmenu, FG_ACTION, this);
  fgElement_AddDelegateListener<LayoutTab, &LayoutTab::MenuContext>(_contextmenulayout, FG_ACTION, this);

  fgTransform tfeditbox = { { 0, 0, 0, 0, 50, 0, 0, 0 }, 0, {0,0,0,0} };
  fgTextbox_Init(&_editbox, 0, 0, "Editor$layouteditbox", FGELEMENT_EXPANDY | FGTEXTBOX_SINGLELINE | FGTEXTBOX_ACTION, &tfeditbox, 0);
  fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_editboxOnAction>(_editbox, FG_ACTION, this);
  fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_editboxOnAction>(_editbox, FG_LOSTFOCUS, this);

  _propafter = [this](fgElement* e, const char* type) {
    if(!STRICMP(type, "textbox") || !STRICMP(type, "combobox"))
    {
      fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_textboxOnAction>(e, FG_ACTION, this);
      fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_textboxOnAction>(e, FG_LOSTFOCUS, this);
    }
  };

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
    AddProp("Alpha", EditorBase::PROP_ALPHA);
  }
  
  EditorBase::AddMenuControls("LayoutContext$add");
  EditorBase::AddMenuControls("LayoutContext$insert");
  EditorBase::AddMenuControls("SublayoutContext$add");
}

void LayoutTab::_treeItemOnHover(struct _FG_ELEMENT* self, const FG_Msg* m)
{
  fgElement* e = _layoutmap[(fgClassLayout*)self->userdata];
  if(e)
    _base->hoverelement = e;
}

void LayoutTab::AddProp(const char* name, FG_UINT id, const char* type)
{
  fgFlag flags = FGELEMENT_EXPANDY;
  fgMessage f = nullptr;
  if(!STRICMP(type, "textbox"))
    flags |= FGTEXTBOX_ACTION | FGTEXTBOX_SINGLELINE;
  _propafter(_base->AddProp(*_layoutprops, name, type, id, f, flags), type);
}
void LayoutTab::_doInsert(const char* type, bool insert)
{
  fgLayout* display = FindParentLayout(_selected);
  if(_selected->userid == 1)
  {
    auto p = (fgLayout*)_selected->userdata;
    if(!insert)
      p->AddChild(type, type, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0, 0);
    fgElement_ClearType(_selected, "treeitem");
    _openSublayout(_selected, p);
    _openLayout(_selected, p->children);
  }
  else
  {
    auto p = (fgClassLayout*)_selected->userdata;
    if(!insert)
      p->AddChild(type, type, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0, 0);
    fgElement_ClearType(_selected, "treeitem");
    _openLayout(_selected, p->children);
  }
  _base->DisplayLayout(display);
}
void LayoutTab::MenuInsert(struct _FG_ELEMENT*, const FG_Msg* m)
{
  if(m->e && m->e->userdata)
  {
    _doInsert((const char*)m->e->userdata, true);
  }
}
void LayoutTab::MenuAdd(struct _FG_ELEMENT*, const FG_Msg* m)
{
  if(m->e && m->e->userdata)
  {
    _doInsert((const char*)m->e->userdata, false);
  }
}
void LayoutTab::MenuContext(struct _FG_ELEMENT*, const FG_Msg* m)
{
  switch(m->e->userid)
  {
  case 1: // insert layout
    if(_selected && _selected->userid == 0)
      break; // Can't insert layouts on non-layout elements
    _editbox->userdata = !_selected ? &_base->curlayout : _selected->userdata;
    _editbox->SetParent(!_selected ? *_layoutview : _selected, 0);
    {
      AbsRect r;
      ResolveRect(_editbox, &r);
      fgSendSubMsg<FG_ACTION, void*>(*_layoutview, FGSCROLLBAR_SCROLLTOABS, &r);
    }
    _editbox->GotFocus();
    break;
  case 4: // remove
    if(_selected && _selected->userdata && _selected->parent && _selected->parent->userdata)
    {
      if(_selected->userid == 1)
      {
        assert(_selected->parent->userid == 1);
        auto p = (fgLayout*)_selected->parent->userdata;
        p->RemoveLayout(((fgLayout*)_selected->userdata)->base.name);
        VirtualFreeChild(_selected);
      }
      else if(_selected->parent->userid == 1)
      {
        fgElement* parent = _selected->parent;
        auto p = (fgLayout*)parent->userdata;
        p->RemoveChild((fgClassLayout*)_selected->userdata - p->children.p);
        fgElement_ClearType(parent, "treeitem");
        _openSublayout(parent, p);
        _openLayout(parent, p->children);
      }
      else
      {
        fgElement* parent = _selected->parent;
        auto p = (fgClassLayout*)parent->userdata;
        p->RemoveChild((fgClassLayout*)_selected->userdata - p->children.p);
        fgElement_ClearType(parent, "treeitem");
        _openLayout(parent, p->children);
      }
      _base->DisplayLayout(&_base->curlayout);
    }
    break;
  }
}
void LayoutTab::_treeviewOnMouseDown(struct _FG_ELEMENT* e, const FG_Msg* msg)
{
  _selected = 0;
  (*_layoutprops)->GotFocus();
}

void LayoutTab::_textboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg)
{
  if(!msg->subtype && _selected && _selected->userdata)
  {
    switch(_selected->userid)
    {
    case 0:
    {
      auto p = (fgClassLayout*)_selected->userdata;
      _base->ParseStyleMsg(p->element.style, _layoutmap[p], &p->element, p, &p->element.skin, &p->name, EditorBase::PROPERTIES(e->userid), e->GetText());
      _base->SetProps(*_layoutprops, p, &p->element, p->element.skin, p->name, p->element.style);
    }
      break;
    case 1:
    {
      auto p = (fgLayout*)_selected->userdata;
      _base->ParseStyleMsg(p->base.style, 0, 0, 0, &p->skin, &p->base.name, EditorBase::PROPERTIES(e->userid), e->GetText());
      _base->SetProps(*_layoutprops, 0, 0, p->skin, p->base.name, p->base.style);
    }
    break;
    }
  }
}

void LayoutTab::_editboxOnAction(struct _FG_ELEMENT* e, const FG_Msg* msg)
{
  if(!msg->subtype)
  {
    const char* name = e->GetText();
    if(e->userdata && name != 0 && name[0] != 0)
    {
      auto p = (fgLayout*)e->userdata;
      e->userdata = 0;
      p->AddLayout(name);
      fgElement_ClearType(e->parent, "treeitem");
      _openSublayout(e->parent, p);
      _openLayout(e->parent, p->children);
    }
    e->SetParent(0);
    e->SetText(0);
  }
}

void LayoutTab::_openLayout(fgElement* root, const fgVectorClassLayout& layout)
{
  for(size_t i = 0; i < layout.l; ++i)
  {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = layout.p + i;
    _treemap.Insert(layout.p + i, item);
    fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_treeItemOnHover>(item, FG_HOVER, this);
    fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_treeItemOnFocus>(item, FG_GOTFOCUS, this);
    item->SetContextMenu(_contextmenu);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(layout.p[i].element.type);
    _openLayout(item, layout.p[i].children);
  }
}
void LayoutTab::_openSublayout(fgElement* root, fgLayout* parent)
{
  std::function<void(fgLayout*, const char*)> fn = [this, root](fgLayout* l, const char* s) {
    fgElement* item = fgCreate("TreeItem", root, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    item->userdata = l;
    item->userid = 1;
    fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_treeItemOnFocus>(item, FG_GOTFOCUS, this);
    item->SetContextMenu(_contextmenulayout);
    fgCreate("Text", item, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0)->SetText(s);
    _openSublayout(item, l);
    _openLayout(item, l->children);
  };

  fgLayout_IterateLayouts(parent, &fn, &Delegate<void, fgLayout*, const char*>::stublambda);
}
void LayoutTab::OpenLayout(fgLayout* layout)
{
  if(_layoutview)
  {
    fgElement_Clear(*_layoutview);

    fgElement* root = fgCreate("Text", *_layoutview, 0, 0, FGELEMENT_EXPAND, &fgTransform_EMPTY, 0);
    root->SetText("root");
    fgElement_AddDelegateListener<LayoutTab, &LayoutTab::_treeviewOnMouseDown>(root, FG_MOUSEDOWN, this);
    (*_layoutview)->userdata = layout;
    (*_layoutview)->userid = 1;
    (*_layoutview)->SetContextMenu(_contextmenulayout);
    _openSublayout(*_layoutview, layout);
    _openLayout(*_layoutview, layout->children);
  }
}
void LayoutTab::_addLayoutSkins(fgSkinBase* base, fgElement* target)
{
  if(!base)
    return;
  fgSkinBase_IterateSkins(base, target, [](void* p, fgSkin* s, const char* n) { fgCreate("text", (fgElement*)p, 0, 0, FGELEMENT_EXPAND, 0, 0)->SetText(n); });
  _addLayoutSkins(base->parent, target);
}
void LayoutTab::_treeItemOnFocus(struct _FG_ELEMENT* e, const FG_Msg*)
{
  if(e->userdata)
  {
    if(e->userid == 1)
    {
      auto p = (fgLayout*)e->userdata;
      _base->LoadProps(*_layoutprops, 0, 0, 0, p->skin, p->base.name, p->base.style, _propafter);
    }
    else
    {
      auto p = (fgClassLayout*)e->userdata;
      _base->LoadProps(*_layoutprops, p->element.type, p, &p->element, p->element.skin, p->name, p->element.style, _propafter);
      if(fgElement* e = _layoutmap[p])
        _base->SetCurElement(e);
    }
    _selected = e;
    fgLayout* display = FindParentLayout(e);
    if(display)
    {
      _base->DisplayLayout(display);
      _addLayoutSkins(&display->base, _layoutprops->GetRow(3)->GetItem(1));
    }
  }
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
