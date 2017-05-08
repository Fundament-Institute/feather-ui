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
  Clear();
}
void LayoutTab::Init(EditorBase* base)
{
  _base = base;
  _layoutview = reinterpret_cast<fgTreeview*>(fgGetID("Editor$layout"));
  _layoutprops = reinterpret_cast<fgGrid*>(fgGetID("Editor$layoutprops"));
  fgTextbox_Init(&_editbox, *_layoutprops, 0, "Editor$editbox", FGELEMENT_BACKGROUND | FGELEMENT_EXPANDY, &fgTransform_EMPTY, 0);

  if(_layoutprops)
  {
    AddProp("(Type)", 0);
    AddProp("ID", EditorBase::PROP_ID);
    AddProp("Name", EditorBase::PROP_NAME);
    AddProp("Skin", EditorBase::PROP_SKIN, "combobox");
    AddProp("Transform", 0);
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
    AddProp("Flags", EditorBase::PROP_FLAGS);
    AddProp("Alpha", EditorBase::PROP_ALPHA, "slider");
    (*_layoutprops)->userdata = this;
    (*_layoutprops)->AddListener(FG_ACTION, _layoutPropsAction);
  }
}
void LayoutTab::_layoutPropsAction(fgElement* e, const FG_Msg* m)
{
  auto p = (LayoutTab*)e->userdata;
  switch(m->subtype)
  {
  case FGSCROLLBAR_BAR:
  case FGSCROLLBAR_PAGE:
  case FGSCROLLBAR_BUTTON:
  case FGSCROLLBAR_CHANGE:
  case FGSCROLLBAR_SCROLLTO:
  case FGSCROLLBAR_SCROLLTOABS:
    if(!(p->_editbox->flags&FGELEMENT_HIDDEN))
    {
      CRect area = p->_editbox->transform.area;
      area.top.abs += e->padding.top - p->_lastscroll.y;
      area.bottom.abs += e->padding.top - p->_lastscroll.y;
      p->_editbox->SetArea(area);
      p->_lastscroll = e->padding.topleft;
    }
  }
}

size_t LayoutTab::_propertyMessage(fgText* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEUP:
    if(self->element.parent && self->element.parent->parent)
    {
      auto p = (LayoutTab*)self->element.parent->parent->userdata;
      AbsRect rect;
      fgElement_RelativeTo(&self->element, self->element.parent->parent, &rect);
      CRect area = { rect.left, 0, rect.top, 0, rect.right, 0, rect.bottom, 0 };
      if(p->_editbox->userdata)
        ((fgElement*)p->_editbox->userdata)->SetFlag(FGELEMENT_HIDDEN, false);
      p->_lastscroll = self->element.parent->parent->padding.topleft;
      p->_editbox->SetArea(area);
      p->_editbox->SetText(self->element.GetText());
      p->_editbox->SetFlag(FGELEMENT_HIDDEN, false);
      p->_editbox->userdata = self;
      p->_editbox->userid = self->element.userid;
      self->element.SetFlag(FGELEMENT_HIDDEN, true);
    }
    break;
  }

  return fgText_Message(self, msg);
}

void LayoutTab::AddProp(const char* name, FG_UINT id, const char* type)
{
  _base->AddProp(*_layoutprops, name, type, id, (!STRICMP(type, "text")) ? (fgMessage)_propertyMessage : nullptr);
}

size_t LayoutTab::_editboxMessage(fgTextbox* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;
  switch(msg->type)
  {
    //case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    //  otherint = bss::bssSetBit<fgFlag>(self->scroll->flags, otherint, msg->u2 != 0);
    //case FG_SETFLAGS:
    //  if((self->scroll->flags ^ otherint) & FGELEMENT_HIDDEN)
    //  {
    //    if(self->scroll->userdata && (otherint & FGELEMENT_HIDDEN))
    //    {
    //      ((fgElement*)self->scroll->userdata)->SetFlag(FGELEMENT_HIDDEN, false);
    //      self->scroll->userdata = 0;
    //    }
    //  }
    //  break;
  case FG_ACTION:
    if(!msg->subtype)
    {
      //fgElement* e = (fgElement*)((fgElement*)fgdebug_instance->elements->userdata)->userdata;
      //fgDebug_ApplyProperty(e, self->scroll->userid, self->scroll->GetText());
      //self->scroll->SetFlag(FGELEMENT_HIDDEN, true);
      //fgDebug_DisplayProperties(e);
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
    _selected = (fgTreeItem*)e;
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
}
