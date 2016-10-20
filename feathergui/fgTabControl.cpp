// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTabControl.h"
#include "feathercpp.h"
#include "fgRadioButton.h"

void FG_FASTCALL fgTabControl_Init(fgTabControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgTabControl_Destroy, (fgMessage)&fgTabControl_Message);
}

void FG_FASTCALL fgTabControl_Destroy(fgTabControl* self)
{
  fgBox_Destroy(&self->header);
  fgControl_Destroy(&self->control);
}

size_t FG_FASTCALL fgTabControlToggle_Message(fgRadiobutton* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_SETVALUE:
    if(msg->subtype > FGVALUE_INT64)
        return 0;
    ((fgElement*)self->window->userdata)->SetFlag(FGELEMENT_HIDDEN, !msg->otherint);
    break;
  case FG_GETSELECTEDITEM:
    return (size_t)self->window->userdata;
  }
  return fgRadiobutton_Message(self, msg);
}

size_t FG_FASTCALL fgTabControlPanel_Message(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_GETSELECTEDITEM:
    return (size_t)self->userdata;
  }
  return fgElement_Message(self, msg);
}

void FG_FASTCALL fgTabControlToggle_Destroy(fgRadiobutton* self)
{
  if(self->window->userdata != 0)
  {
    fgElement* panel = (fgElement*)self->window->userdata;
    self->window->userdata = 0; // prevents an infinite loop
    panel->userdata = 0; // prevents deleting the panel or the toggle twice
    VirtualFreeChild(panel);
  }
  fgRadiobutton_Destroy(self);
}

void FG_FASTCALL fgTabControlPanel_Destroy(fgElement* self)
{
  if(self->userdata != 0)
  {
    fgElement* toggle = (fgElement*)self->userdata;
    self->userdata = 0; // prevents an infinite loop
    toggle->userdata = 0; // prevents deleting the panel or the toggle twice
    VirtualFreeChild(toggle);
  }
  fgElement_Destroy(self);
}

size_t FG_FASTCALL fgTabControl_Message(fgTabControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgBox_Init(&self->header, *self, 0, "fgTabControl$header", FGBOX_TILEX | FGELEMENT_EXPANDY | FGELEMENT_BACKGROUND, &fgTransform { 0, 0, 0, 0, 0, 1, 0, 0 });
    return FG_ACCEPT;
  case FG_ADDITEM:
  {
    fgElement* button = fgCreate("RadioButton", self->header, 0, "fgTabControl$toggle", FGELEMENT_EXPAND, &fgTransform_EMPTY);
    assert(button->destroy == (fgDestroy)&fgRadiobutton_Destroy);
    fgElement* panel = fgCreate("Element", *self, 0, "fgTabControl$panel", FGELEMENT_HIDDEN, &fgTransform_DEFAULT);
    assert(panel->message == (fgMessage)&fgElement_Message);
    panel->message = (fgMessage)&fgTabControlPanel_Message;
    panel->destroy = (fgDestroy)&fgTabControlPanel_Destroy;
    button->message = (fgMessage)&fgTabControlToggle_Message;
    button->destroy = (fgDestroy)&fgTabControlToggle_Destroy;
    panel->userdata = button;
    button->userdata = panel;
    button->SetText((const char*)msg->other);
    AbsRect padding = self->control->padding;
    padding.top = self->header->transform.area.bottom.abs;
    self->control->SetPadding(padding);
    button->SetValue(1);
    return (size_t)panel;
  }
  case FG_GETSELECTEDITEM:
    return (size_t)self->selected;
  case FG_GETCLASSNAME:
    return (size_t)"TabControl";
  }

  return fgControl_HoverMessage(&self->control, msg);
}

fgElement* fgTabControl::AddItem(const char* name) { return (fgElement*)_sendmsg<FG_ADDITEM, const void*>(*this, name); }
