// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCombobox.h"
#include "feathercpp.h"

void fgCombobox_Init(fgCombobox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgCombobox_Destroy, (fgMessage)&fgCombobox_Message);
}
void fgCombobox_Destroy(fgCombobox* self)
{
  self->dropdown->message = (fgMessage)fgDropdown_Message;
  fgDropdown_Destroy(&self->dropdown);
}

size_t fgComboboxDropdown_Message(fgBox* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEUP:
    fgDropdownBox_Message(self, msg);
    if(self->scroll->flags&FGELEMENT_HIDDEN)
      self->scroll->parent->SetText(reinterpret_cast<fgDropdown*>(self->scroll->parent)->selected->GetText());
    return FG_ACCEPT;
  }

  return fgDropdownBox_Message(self, msg);
}
size_t fgCombobox_Message(fgCombobox* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgDropdown_Message(&self->dropdown, msg);
    fgTextbox_Init(&self->text, *self, 0, "fgCombobox$textbox", FGTEXTBOX_SINGLELINE | FGTEXTBOX_ACTION | FGELEMENT_BACKGROUND | FGSCROLLBAR_HIDEV | FGSCROLLBAR_HIDEH, &fgTransform_DEFAULT, 0);
    self->dropdown.box->message = (fgMessage)fgComboboxDropdown_Message;
    return FG_ACCEPT;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETLINEHEIGHT:
  case FG_SETLETTERSPACING:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETLINEHEIGHT:
  case FG_GETLETTERSPACING:
  case FG_GETCOLOR:
    return fgSendMessage(self->text, msg);
  case FG_GETSELECTEDITEM:
    return (size_t)&self->text;
  case FG_DRAW:
    return fgControl_Message(&self->dropdown.control, msg);
  case FG_GETCLASSNAME:
    return (size_t)"Combobox";
  }

  return fgDropdown_Message(&self->dropdown, msg);
}