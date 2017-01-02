// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCheckbox.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgCheckbox_Destroy, (fgMessage)&fgCheckbox_Message);
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
    fgText_Init(&self->text, *self, 0, "Checkbox$text", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER, 0);
    fgMaskSetStyle(*self, "default", fgStyleGetMask("default", "checked", "indeterminate"));
    self->checked = FGCHECKED_NONE;
    return FG_ACCEPT;
  case FG_NEUTRAL:
    fgStandardNeutralSetStyle(*self, "neutral");
    return FG_ACCEPT;
  case FG_HOVER:
    fgStandardNeutralSetStyle(*self, "hover");
    return FG_ACCEPT;
  case FG_ACTIVE:
    fgStandardNeutralSetStyle(*self, "active");
    return FG_ACCEPT;
  case FG_ACTION:
    fgIntMessage(*self, FG_SETVALUE, !_sendmsg<FG_GETVALUE>(*self), 0);
    return FG_ACCEPT;
  case FG_SETVALUE:
    if(msg->subtype != 0 && msg->subtype != FGVALUE_INT64)
      return 0;
    self->checked = (char)msg->otherint;
    fgMaskSetStyle(*self, (self->checked == FGCHECKED_CHECKED) ? "checked" : ((self->checked == FGCHECKED_INDETERMINATE) ? "indeterminate" : "default"), fgStyleGetMask("default", "checked", "indeterminate"));
    return FG_ACCEPT;
  case FG_GETVALUE:
    if(!msg->subtype || msg->subtype == FGVALUE_INT64)
      return self->checked;
    return 0;
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