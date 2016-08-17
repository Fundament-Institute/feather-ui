// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgButton_Destroy, (fgMessage)&fgButton_Message);
}
void FG_FASTCALL fgButton_Destroy(fgButton* self)
{
  fgControl_Destroy((fgControl*)self);
}
size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  ptrdiff_t otherint = msg->otherint;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgText_Init(&self->text, *self, 0, "Button:text", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER);
    fgElement_Init(&self->item, *self, self->text, "Button:item", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER);
    fgStandardNuetralSetStyle(*self, (self->control.element.flags & FGCONTROL_DISABLE) ? "disabled" : "nuetral");
    return FG_ACCEPT;
  case FG_ADDITEM:
    if(!msg->other)
      fgElement_Clear(&self->item);
    else
      fgPassMessage(&self->item, msg);
    return FG_ACCEPT;
  case FG_NUETRAL:
    fgStandardNuetralSetStyle(*self, "nuetral");
    return FG_ACCEPT;
  case FG_HOVER:
    fgStandardNuetralSetStyle(*self, "hover");
    return FG_ACCEPT;
  case FG_ACTIVE:
    fgStandardNuetralSetStyle(*self, "active");
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->control.element.flags&FGBUTTON_NOFOCUS)
      return 0;
    break;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->control.element.flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
    if((self->control.element.flags ^ (fgFlag)otherint) & FGCONTROL_DISABLE)
    {
      fgStandardNuetralSetStyle(*self, (otherint & FGCONTROL_DISABLE) ? "disabled" : "nuetral");
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