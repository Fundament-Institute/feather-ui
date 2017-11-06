// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgCombobox.h"

void fgCombobox_Init(fgCombobox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgCombobox_Destroy, (fgMessage)&fgCombobox_Message);
}
void fgCombobox_Destroy(fgCombobox* self)
{
  self->dropdown->message = (fgMessage)fgDropdown_Message;
  fgDropdown_Destroy(&self->dropdown);
}

size_t fgComboboxTextbox_Message(fgTextbox* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ACTION:
    if(!msg->subtype && (*self)->parent)
      return fgSendMessage((*self)->parent, msg);
    break;
  case FG_LOSTFOCUS:
    if((*self)->parent)
    {
      fgTextbox_Message(self, msg);
      fgroot_instance->fgFocusedWindow = (*self)->parent;
      return fgSendMessage((*self)->parent, msg);
    }
    break;
  }

  return fgTextbox_Message(self, msg);
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
  {
    fgDropdown_Message(&self->dropdown, msg);
    fgTransform tf = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    fgTextbox_Init(&self->text, *self, 0, "fgCombobox$textbox", FGTEXTBOX_SINGLELINE | FGTEXTBOX_ACTION | FGELEMENT_EXPANDY | FGSCROLLBAR_HIDEV | FGSCROLLBAR_HIDEH | FGFLAGS_INTERNAL, &tf, 0);
    self->text->message = (fgMessage)fgComboboxTextbox_Message;
    self->dropdown.box->message = (fgMessage)fgComboboxDropdown_Message;
  }
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
  case FG_REMOVECHILD:
  case FG_ADDCHILD:
  case FG_REORDERCHILD:
    if(msg->e == self->text)
      return fgControl_Message(&self->dropdown.control, msg);
    break;
  case FG_GOTFOCUS:
    return fgSendMessage(self->text, msg);
  case FG_LAYOUTCHANGE:
    msg = msg;
    break;
  case FG_CLEAR:
    fgSendMessage(self->text, msg);
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Combobox";
  }

  return fgDropdown_Message(&self->dropdown, msg);
}