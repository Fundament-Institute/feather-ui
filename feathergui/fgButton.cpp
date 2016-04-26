// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT prev, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, flags, parent, prev, transform, (FN_DESTROY)&fgButton_Destroy, (FN_MESSAGE)&fgButton_Message);
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
    fgElement_Init(&self->item, FGELEMENT_EXPAND | FGELEMENT_IGNORE, *self, 0, &fgTransform_CENTER);
    fgElement_AddPreChild(*self, &self->item);
    fgText_Init(&self->text, 0, 0, 0, FGELEMENT_EXPAND | FGELEMENT_IGNORE, *self, 0, &fgTransform_CENTER);
    fgElement_AddPreChild(*self, self->text);
    _sendmsg<FG_SETSTYLE, void*>(*self, "nuetral");
    return 1;
  case FG_ADDITEM:
    if(msg->other)
      _sendmsg<FG_ADDCHILD, void*>(&self->item, msg->other);
    else
      fgElement_Clear(&self->item);
    return 1;
  case FG_NUETRAL:
    _sendmsg<FG_SETSTYLE, void*>(*self, "nuetral");
    return 1;
  case FG_HOVER:
    _sendmsg<FG_SETSTYLE, void*>(*self, "hover");
    return 1;
  case FG_ACTIVE:
    _sendmsg<FG_SETSTYLE, void*>(*self, "active");
    return 1;
  case FG_GOTFOCUS:
    if(self->control.element.flags&FGBUTTON_NOFOCUS)
      return 0;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgButton";
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgPassMessage(self->text, msg);
  }
  return fgControl_HoverMessage(&self->control, msg);
}