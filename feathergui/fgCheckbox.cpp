// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCheckbox.h"
#include "fgSkin.h"
#include "feathercpp.h"

FG_EXTERN void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgFlag flags, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT prev, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, flags, parent, prev, transform, (FN_DESTROY)&fgCheckbox_Destroy, (FN_MESSAGE)&fgCheckbox_Message);
}
FG_EXTERN void FG_FASTCALL fgCheckbox_Destroy(fgCheckbox* self)
{
  fgControl_Destroy(&self->control);
}
FG_EXTERN size_t FG_FASTCALL fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgElement_Init(&self->check, FGELEMENT_EXPAND | FGELEMENT_IGNORE | FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN, *self, 0, 0);
    fgElement_AddPreChild(*self, &self->check);
    fgElement_Init(&self->indeterminate, FGELEMENT_EXPAND | FGELEMENT_IGNORE | FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN, *self, 0, 0);
    fgElement_AddPreChild(*self, &self->indeterminate);
    fgText_Init(&self->text, 0, 0, 0, FGELEMENT_EXPAND | FGELEMENT_IGNORE, *self, 0, &fgTransform_CENTER);
    fgElement_AddPreChild(*self, self->text);
    fgElement_Init(&self->item, FGELEMENT_EXPAND | FGELEMENT_IGNORE, *self, 0, &fgTransform_CENTER);
    fgElement_AddPreChild(*self, &self->item);
    _sendmsg<FG_SETSTYLE, void*>(*self, "nuetral");
    self->checked = 0;
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
  case FG_ACTION:
    fgIntMessage(*self, FG_SETSTATE, !_sendmsg<FG_GETSTATE>(*self), 0);
    return 1;
  case FG_SETSTATE:
    self->checked = msg->otherint;
    fgIntMessage(&self->check, FG_SETFLAG, FGELEMENT_HIDDEN, self->checked != 1);
    fgIntMessage(&self->indeterminate, FG_SETFLAG, FGELEMENT_HIDDEN, self->checked != 2);
    return 1;
  case FG_GETSTATE:
    return self->checked;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgPassMessage(self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"fgCheckbox";
  }
  return fgControl_HoverMessage(&self->control, msg);
}