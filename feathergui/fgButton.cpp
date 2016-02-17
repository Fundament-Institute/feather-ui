// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, prev, element, (FN_DESTROY)&fgButton_Destroy, (FN_MESSAGE)&fgButton_Message);
}
void FG_FASTCALL fgButton_Destroy(fgButton* self)
{
  fgWindow_Destroy((fgWindow*)self);
}
size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_HoverProcess(&self->window, msg);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, 0, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->text);
    fgChild_Init(&self->item, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, 0, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->item);
    fgChild_VoidMessage((fgChild*)self, FG_SETSTYLE, "nuetral");
    return 0;
  case FG_ADDITEM:
    if(msg->other)
      fgChild_VoidMessage(&self->item, FG_ADDCHILD, msg->other);
    else
      fgChild_Clear(&self->item);
    return 0;
  case FG_NUETRAL:
    fgChild_VoidMessage((fgChild*)self, FG_SETSTYLE, "nuetral");
    return 0;
  case FG_HOVER:
    fgChild_VoidMessage((fgChild*)self, FG_SETSTYLE, "hover");
    return 0;
  case FG_ACTIVE:
    fgChild_VoidMessage((fgChild*)self, FG_SETSTYLE, "active");
    return 0;
  case FG_GOTFOCUS:
    if(self->window.element.flags&FGBUTTON_NOFOCUS)
      return 1;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgButton";
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage((fgChild*)&self->text, msg);
  }
  return fgWindow_HoverProcess(&self->window,msg);
}