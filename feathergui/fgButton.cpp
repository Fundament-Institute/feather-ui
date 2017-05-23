// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void fgButton_Init(fgButton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgButton_Destroy, (fgMessage)&fgButton_Message);
}
void fgButton_Destroy(fgButton* self)
{
  self->control->message = (fgMessage)fgControl_Message;
  fgControl_Destroy((fgControl*)self);
}
size_t fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgText_Init(&self->text, *self, 0, "Button$text", FGELEMENT_EXPAND | FGELEMENT_IGNORE | FGFLAGS_INTERNAL, &fgTransform_CENTER, 0);
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgButton* hold = reinterpret_cast<fgButton*>(msg->e);
      fgControl_ActionMessage(&self->control, msg);
      self->text->Clone(hold->text);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->text);
    }
    return sizeof(fgButton);
  case FG_REORDERCHILD:
    if(msg->e != (fgElement*)&self->text && !msg->e2)
    {
      FG_Msg m = *msg;
      m.p2 = &self->text;
      return fgControl_ActionMessage(&self->control, &m);
    }
    break;
  case FG_ADDCHILD:
  case FG_ADDITEM:
    if(msg->e != (fgElement*)&self->text && !msg->e2)
    {
      FG_Msg m = *msg;
      m.p2 = &self->text;
      return fgControl_ActionMessage(&self->control, &m);
    }
    break;
  case FG_SETITEM:
    if(msg->subtype == FGITEM_TEXT || msg->subtype == FGITEM_DEFAULT)
      return self->control->SetText((const char*)msg->p, (FGTEXTFMT)msg->u2);
    break;
  case FG_GOTFOCUS:
    if(self->control.element.flags&FGBUTTON_NOFOCUS)
      return 0;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Button";
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
  }
  return fgControl_ActionMessage(&self->control, msg);
};