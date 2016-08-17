// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCheckbox.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgCheckbox_Destroy, (fgMessage)&fgCheckbox_Message);
}
void FG_FASTCALL fgCheckbox_Destroy(fgCheckbox* self)
{
  fgControl_Destroy(&self->control);
}
size_t FG_FASTCALL fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgText_Init(&self->text, *self, 0, "Checkbox:text", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER);
    fgMaskSetStyle(*self, "default", fgStyleGetMask("default", "checked", "indeterminate"));
    self->checked = 0;
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
  case FG_ACTION:
    fgIntMessage(*self, FG_SETSTATE, !_sendmsg<FG_GETSTATE>(*self), 0);
    return FG_ACCEPT;
  case FG_SETSTATE:
    self->checked = msg->otherint;
    fgMaskSetStyle(*self, (self->checked == 1) ? "checked" : ((self->checked == 2) ? "indeterminate" : "default"), fgStyleGetMask("default", "checked", "indeterminate"));
    return FG_ACCEPT;
  case FG_GETSTATE:
    return self->checked;
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
  case FG_GETCLASSNAME:
    return (size_t)"Checkbox";
  }
  return fgControl_ActionMessage(&self->control, msg);
}