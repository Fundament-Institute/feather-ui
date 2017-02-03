// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void fgButton_Init(fgButton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
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
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgText_Init(&self->text, *self, 0, "Button$text", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER, 0);
    fgStandardNeutralSetStyle(*self, (self->control.element.flags & FGCONTROL_DISABLE) ? "disabled" : "neutral");
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
  case FG_ADDCHILD:
  case FG_ADDITEM:
    if(msg->p != (fgElement*)&self->text && !msg->p2)
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
  case FG_NEUTRAL:
    fgStandardNeutralSetStyle(*self, "neutral");
    return FG_ACCEPT;
  case FG_HOVER:
    fgStandardNeutralSetStyle(*self, "hover");
    return FG_ACCEPT;
  case FG_ACTIVE:
    fgStandardNeutralSetStyle(*self, "active");
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->control.element.flags&FGBUTTON_NOFOCUS)
      return 0;
    break;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->control.element.flags, otherint, msg->u2);
  case FG_SETFLAGS:
    if((self->control.element.flags ^ (fgFlag)otherint) & FGCONTROL_DISABLE)
    {
      fgStandardNeutralSetStyle(*self, (otherint & FGCONTROL_DISABLE) ? "disabled" : "neutral");
      fgroot_instance->mouse.state |= FGMOUSE_SEND_MOUSEMOVE;
    }
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
    return fgPassMessage(self->text, msg);
  }
  return fgControl_ActionMessage(&self->control, msg);
}