// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCheckbox.h"
#include "fgSkin.h"
#include "feathercpp.h"

FG_EXTERN void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgCheckbox_Destroy, (FN_MESSAGE)&fgCheckbox_Message);
}
FG_EXTERN void FG_FASTCALL fgCheckbox_Destroy(fgCheckbox* self)
{
  fgWindow_Destroy(&self->window);
}
FG_EXTERN size_t FG_FASTCALL fgCheckbox_Message(fgCheckbox* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_HoverMessage(&self->window, msg);
    fgChild_Init(&self->check, FGCHILD_EXPAND | FGCHILD_IGNORE | FGCHILD_BACKGROUND, *self, 0, 0);
    fgChild_AddPreChild(*self, &self->check);
    fgChild_Init(&self->indeterminate, FGCHILD_EXPAND | FGCHILD_IGNORE | FGCHILD_BACKGROUND, *self, 0, 0);
    fgChild_AddPreChild(*self, &self->indeterminate);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, *self, 0, &fgElement_CENTER);
    fgChild_AddPreChild(*self, self->text);
    fgChild_Init(&self->item, FGCHILD_EXPAND | FGCHILD_IGNORE, *self, 0, &fgElement_CENTER);
    fgChild_AddPreChild(*self, &self->item);
    fgSendMsg<FG_SETSTYLE, void*>(*self, "nuetral");
    self->checked = 0;
    return 1;
  case FG_ADDITEM:
    if(msg->other)
      fgSendMsg<FG_ADDCHILD, void*>(&self->item, msg->other);
    else
      fgChild_Clear(&self->item);
    return 1;
  case FG_NUETRAL:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "nuetral");
    return 1;
  case FG_HOVER:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "hover");
    return 1;
  case FG_ACTIVE:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "active");
    return 1;
  case FG_ACTION:
    fgChild_IntMessage(*self, FG_SETSTATE, !fgSendMsg<FG_GETSTATE>(*self), 0);
    return 1;
  case FG_SETSTATE:
    self->checked = msg->otherint;
    fgChild_IntMessage(&self->check, FG_SETFLAG, FGCHILD_HIDDEN, self->checked != 1);
    fgChild_IntMessage(&self->indeterminate, FG_SETFLAG, FGCHILD_HIDDEN, self->checked != 2);
    return 1;
  case FG_GETSTATE:
    return self->checked;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage(self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"fgCheckbox";
  }
  return fgWindow_HoverMessage(&self->window, msg);
}