// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "fgText.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self,flags,parent,element, &fgButton_Destroy, &fgButton_Message);
}
void FG_FASTCALL fgButton_Destroy(fgButton* self)
{
  assert(self != 0);
  fgText_Destroy(&self->text);
  fgChild_Destroy(&self->item);
  fgWindow_Destroy((fgWindow*)self);
}
size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  static const fgElement defelem = { 0,0,0,0,0,0,0,0,0,0,0.5f,0,0.5f };

  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_HoverProcess(&self->window, msg);
    self->state = 0;
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_BACKGROUND | FGCHILD_IGNORE, (fgChild*)self, &fgElement_DEFAULT);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->text);
    fgChild_Init(&self->item, FGCHILD_EXPAND, (fgChild*)self, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->item);
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