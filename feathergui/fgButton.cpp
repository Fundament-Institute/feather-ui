// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "feathercpp.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgButton_Destroy, (FN_MESSAGE)&fgButton_Message);
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
    fgChild_Init(&self->item, FGCHILD_EXPAND | FGCHILD_IGNORE, *self, 0, &fgElement_CENTER);
    fgChild_AddPreChild(*self, &self->item);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, *self, 0, &fgElement_CENTER);
    fgChild_AddPreChild(*self, self->text);
    fgSendMsg<FG_SETSTYLE, void*>(*self, "nuetral");
    return 0;
  case FG_ADDITEM:
    if(msg->other)
      fgSendMsg<FG_ADDCHILD, void*>(&self->item, msg->other);
    else
      fgChild_Clear(&self->item);
    return 0;
  case FG_NUETRAL:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "nuetral");
    return 0;
  case FG_HOVER:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "hover");
    return 0;
  case FG_ACTIVE:
    fgSendMsg<FG_SETSTYLE, void*>(*self, "active");
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
    return fgChild_PassMessage(self->text, msg);
  }
  return fgWindow_HoverProcess(&self->window,msg);
}