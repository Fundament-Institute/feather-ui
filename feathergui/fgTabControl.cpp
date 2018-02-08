// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgTabcontrol.h"

void fgTabcontrol_Init(fgTabcontrol* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgTabcontrol_Destroy, (fgMessage)&fgTabcontrol_Message);
}

void fgTabcontrol_Destroy(fgTabcontrol* self)
{
  fgList_Destroy(&self->header);
  self->control->message = (fgMessage)fgControl_HoverMessage;
  fgControl_Destroy(&self->control);
}

size_t fgTabcontrolToggle_Message(fgControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ACTION:
  {
    FG_Msg m = { FG_ACTION, 0 };
    m.e = *self;
    return fgSendMessage(self->element.parent, &m);
  }
  case FG_GETSELECTEDITEM:
    return (size_t)self->element.userdata;
  }
  return fgListItem_Message(self, msg);
}

size_t fgTabcontrolPanel_Message(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ACTION:
    if(self->userdata)
      return fgSendMessage((fgElement*)self->userdata, msg);
    return 0;
  case FG_GETSELECTEDITEM:
    return (size_t)self->userdata;
  }
  return fgElement_Message(self, msg);
}

void fgTabcontrolToggle_Destroy(fgControl* self)
{
  if(self->element.userdata != 0)
  {
    fgElement* panel = (fgElement*)self->element.userdata;
    self->element.userdata = 0; // prevents an infinite loop
    panel->userdata = 0; // prevents deleting the panel or the toggle twice
    VirtualFreeChild(panel);
  }
  self->element.message = (fgMessage)fgText_Message;
  fgControl_Destroy(self);
}

void fgTabcontrolPanel_Destroy(fgElement* self)
{
  if(self->userdata != 0)
  {
    fgElement* toggle = (fgElement*)self->userdata;
    self->userdata = 0; // prevents an infinite loop
    toggle->userdata = 0; // prevents deleting the panel or the toggle twice
    VirtualFreeChild(toggle);
  }
  self->message = (fgMessage)&fgElement_Message;
  fgElement_Destroy(self);
}

void fgTabcontrol_AdjustPadding(fgTabcontrol* self)
{
  AbsRect padding = self->realpadding;
  padding.top += self->header->transform.area.bottom.abs;
  FG_Msg m = { FG_SETPADDING, 0 };
  m.p = &padding;
  fgControl_HoverMessage(&self->control, &m);
}

fgElement* fgTabcontrolHeader_PreSelect(fgList* self)
{
  return (self->box.selected.l > 0) ? self->box.selected.p[0] : 0;
}

void fgTabcontrolHeader_PostSelect(fgList* self, fgElement* selected)
{
  fgElement* n = 0;
  if(self->box.selected.l > 0)
    n = self->box.selected.p[0];
  if(selected != n)
  {
    if(selected)
      ((fgElement*)selected->userdata)->SetFlag(FGELEMENT_HIDDEN, 1);
    if(n)
      ((fgElement*)n->userdata)->SetFlag(FGELEMENT_HIDDEN, 0);
  }
}

size_t fgTabcontrolHeader_Message(fgList* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ACTION:
    if(msg->e)
    {
      assert(msg->e->parent == *self);
      fgElement* selected = fgTabcontrolHeader_PreSelect(self);
      fgBox_DeselectAll(&self->box);
      fgBox_SelectTarget(&self->box, msg->e);
      fgTabcontrolHeader_PostSelect(self, selected);
    }
    break;
  case FG_MOUSEDOWN:
  {
    fgElement* selected = fgTabcontrolHeader_PreSelect(self);
    size_t ret = fgList_Message(self, msg);
    fgTabcontrolHeader_PostSelect(self, selected);
    return ret;
  }
  }
  return fgList_Message(self, msg);
}

size_t fgTabcontrol_Message(fgTabcontrol* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    bss::bssFill(self->realpadding, 0);
    fgControl_HoverMessage(&self->control, msg);
    fgTransform TF_HEADER = { { 0, 0, 0, 0, 0, 1, 0, 0 }, 0,{ 0,0,0,0 } };
    fgList_Init(&self->header, *self, 0, "Tabcontrol$header", FGLIST_SELECT | FGBOX_TILEX | FGELEMENT_EXPANDY | FGELEMENT_BACKGROUND | (self->control->flags&FGLIST_DRAGGABLE) | FGFLAGS_INTERNAL, &TF_HEADER, 0);
    assert(self->header->message == (fgMessage)&fgList_Message);
    self->header->message = (fgMessage)&fgTabcontrolHeader_Message;
    return FG_ACCEPT;
  }
  case FG_CLONE:
    if(msg->e)
    {
      fgTabcontrol* hold = reinterpret_cast<fgTabcontrol*>(msg->e);
      self->header->Clone(hold->header);
      fgControl_HoverMessage(&self->control, msg);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->header);
    }
    return sizeof(fgTabcontrol);
  case FG_MOVE:
    fgControl_HoverMessage(&self->control, msg);
    if((msg->u2 & FGMOVE_PROPAGATE) != 0 && msg->p == &self->header)
      fgTabcontrol_AdjustPadding(self);
    return FG_ACCEPT;
  case FG_ADDITEM:
  {
    fgElement* button = fgroot_instance->backend.fgCreate("ListItem", self->header, 0, "Tabcontrol$toggle", FGELEMENT_EXPAND, 0, 0);
    fgElement* text = fgroot_instance->backend.fgCreate("Text", button, 0, "Tabcontrol$text", FGELEMENT_EXPAND, 0, 0);
    fgElement* panel = fgroot_instance->backend.fgCreate("Element", *self, 0, "Tabcontrol$panel", FGELEMENT_HIDDEN, &fgTransform_DEFAULT, 0);
    assert(button->message == (fgMessage)&fgListItem_Message);
    assert(panel->message == (fgMessage)&fgElement_Message);
    assert(button->destroy == (fgDestroy)&fgControl_Destroy);
    panel->message = (fgMessage)&fgTabcontrolPanel_Message;
    panel->destroy = (fgDestroy)&fgTabcontrolPanel_Destroy;
    button->message = (fgMessage)&fgTabcontrolToggle_Message;
    button->destroy = (fgDestroy)&fgTabcontrolToggle_Destroy;
    panel->userdata = button;
    button->userdata = panel;
    text->SetText((const char*)msg->p);
    if(self->header.box.selected.l > 0)
      ((fgElement*)self->header.box.selected.p[0]->userdata)->SetFlag(FGELEMENT_HIDDEN, 1);
    self->header.box.selected.l = 0;
    ((fgElementArray&)self->header.box.selected).Insert(button);
    return (size_t)panel;
  }
  case FG_SETPADDING:
    if(msg->p != nullptr)
    {
      AbsRect* padding = (AbsRect*)msg->p;
      char diff = CompareMargins(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff) // Only send a move message if the padding change actually changed something
        fgSubMessage(*self, FG_MOVE, FG_SETPADDING, 0, FGMOVE_PADDING);

      fgTabcontrol_AdjustPadding(self);
      return FG_ACCEPT;
    }
    fgLog(FGLOG_INFO, "%s attempted to set NULL padding", fgGetFullName(*self).c_str());
    return 0;
  case FG_GETSELECTEDITEM:
    return fgSendMessage(self->header, msg);
  case FG_GETCLASSNAME:
    return (size_t)"TabControl";
  }

  return fgControl_HoverMessage(&self->control, msg);
}

FG_EXTERN size_t fgTab_Message(fgElement* self, const FG_Msg* msg)
{
  if(msg->type == FG_GETCLASSNAME)
    return (size_t)"Tab";
  return fgElement_Message(self, msg);
}