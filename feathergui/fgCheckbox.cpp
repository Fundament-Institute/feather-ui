// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgCheckbox.h"
#include "fgSkin.h"

FG_EXTERN void FG_FASTCALL fgCheckbox_Init(fgCheckbox* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, element, (FN_DESTROY)&fgCheckbox_Destroy, (FN_MESSAGE)&fgCheckbox_Message);
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
    fgWindow_HoverProcess(&self->window, msg);
    fgChild_Init(&self->check, FGCHILD_EXPAND | FGCHILD_IGNORE | FGCHILD_BACKGROUND, (fgChild*)self, 0);
    fgChild_AddPreChild((fgChild*)self, &self->check);
    fgChild_Init(&self->indeterminate, FGCHILD_EXPAND | FGCHILD_IGNORE | FGCHILD_BACKGROUND, (fgChild*)self, 0);
    fgChild_AddPreChild((fgChild*)self, &self->indeterminate);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, &fgElement_CENTER);
    fgChild_IntMessage((fgChild*)&self->text, FG_SETORDER, 1, 0);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->text);
    fgChild_Init(&self->item, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->item);
    fgChild_IntMessage((fgChild*)self, FG_SETSTYLE, 0, 0);
    self->checked = 0;
    return 0;
  case FG_ADDITEM:
    if(msg->other)
      fgChild_VoidMessage(&self->item, FG_ADDCHILD, msg->other);
    else
      fgChild_Clear(&self->item);
    return 0;
  case FG_NUETRAL:
    fgChild_IntMessage((fgChild*)self, FG_SETSTYLE, 0, 0);
    return 0;
  case FG_HOVER:
    fgChild_IntMessage((fgChild*)self, FG_SETSTYLE, 1, 0);
    return 0;
  case FG_ACTIVE:
    fgChild_IntMessage((fgChild*)self, FG_SETSTYLE, 2, 0);
    return 0;
  case FG_ACTION:
    fgChild_IntMessage((fgChild*)self, FG_SETSTATE, !fgChild_VoidMessage((fgChild*)self, FG_GETSTATE, 0), 0);
    return 0;
  case FG_SETSTATE:
    self->checked = msg->otherint;
    fgChild_IntMessage(&self->check, FG_SETFLAG, FGCHILD_HIDDEN, self->checked != 1);
    fgChild_IntMessage(&self->indeterminate, FG_SETFLAG, FGCHILD_HIDDEN, self->checked != 2);
    return 0;
  case FG_GETSTATE:
    return self->checked;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage((fgChild*)&self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"fgCheckbox";
  }
  return fgWindow_HoverProcess(&self->window, msg);
}