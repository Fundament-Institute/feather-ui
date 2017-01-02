// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTabcontrol.h"
#include "feathercpp.h"
#include "fgRadiobutton.h"

void FG_FASTCALL fgTabcontrol_Init(fgTabcontrol* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgTabcontrol_Destroy, (fgMessage)&fgTabcontrol_Message);
}

void FG_FASTCALL fgTabcontrol_Destroy(fgTabcontrol* self)
{
  fgBox_Destroy(&self->header);
  fgControl_Destroy(&self->control);
}

size_t FG_FASTCALL fgTabcontrolToggle_Message(fgRadiobutton* self, const FG_Msg* msg)
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

size_t FG_FASTCALL fgTabcontrolPanel_Message(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_GETSELECTEDITEM:
    return (size_t)self->userdata;
  }
  return fgElement_Message(self, msg);
}

void FG_FASTCALL fgTabcontrolToggle_Destroy(fgRadiobutton* self)
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

void FG_FASTCALL fgTabcontrolPanel_Destroy(fgElement* self)
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

size_t FG_FASTCALL fgTabcontrol_Message(fgTabcontrol* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    fgControl_HoverMessage(&self->control, msg);
    fgTransform TF_HEADER = { { 0, 0, 0, 0, 0, 1, 0, 0 }, 0,{ 0,0,0,0 } };
    fgBox_Init(&self->header, *self, 0, "Tabcontrol$header", FGBOX_TILEX | FGELEMENT_EXPANDY | FGELEMENT_BACKGROUND, &TF_HEADER, 0);
    return FG_ACCEPT;
  }
  case FG_ADDITEM:
  {
    fgElement* button = fgroot_instance->backend.fgCreate("Radiobutton", self->header, 0, "Tabcontrol$toggle", FGELEMENT_EXPAND, 0, 0);
    assert(button->destroy == (fgDestroy)&fgRadiobutton_Destroy);
    fgElement* panel = fgroot_instance->backend.fgCreate("Element", *self, 0, "Tabcontrol$panel", FGELEMENT_HIDDEN, &fgTransform_DEFAULT, 0);
    assert(panel->message == (fgMessage)&fgElement_Message);
    panel->message = (fgMessage)&fgTabcontrolPanel_Message;
    panel->destroy = (fgDestroy)&fgTabcontrolPanel_Destroy;
    button->message = (fgMessage)&fgTabcontrolToggle_Message;
    button->destroy = (fgDestroy)&fgTabcontrolToggle_Destroy;
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

fgElement* fgTabcontrol::AddItem(const char* name) { return (fgElement*)_sendmsg<FG_ADDITEM, const void*>(*this, name); }
