// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (FN_DESTROY)&fgButton_Destroy, (FN_MESSAGE)&fgButton_Message);
}
void FG_FASTCALL fgButton_Destroy(fgButton* self)
{
  fgControl_Destroy((fgControl*)self);
}
size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgText_Init(&self->text, *self, 0, "fgButton:text", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER);
    fgElement_Init(&self->item, *self, self->text, "fgButton:item", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER);
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "nuetral", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_ADDITEM:
    if(!msg->other)
      fgElement_Clear(&self->item);
    else
      fgPassMessage(&self->item, msg);
    return FG_ACCEPT;
  case FG_NUETRAL:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "nuetral", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_HOVER:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "hover", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_ACTIVE:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "active", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->control.element.flags&FGBUTTON_NOFOCUS)
      return 0;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgButton";
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
  return fgControl_HoverMessage(&self->control, msg);
}