// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"
#include "fgText.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  assert(self!=0);
  memset(self,0,sizeof(fgButton));
  fgWindow_Init(&self->window,flags,parent,element);
  self->window.element.destroy = &fgButton_Destroy;
  self->window.element.message = &fgButton_Message;

  fgChild_Init(&self->item, FGCHILD_EXPAND, (fgChild*)self, &fgElement_CENTER);
  fgChild_AddPreChild((fgChild*)self, &self->item);
}
void FG_FASTCALL fgButton_Destroy(fgButton* self)
{
  assert(self != 0);
  fgChild_Destroy(&self->item);
  fgWindow_Destroy((fgWindow*)self);
}
size_t FG_FASTCALL fgButton_Message(fgButton* self, const FG_Msg* msg)
{
  static const fgElement defelem = { 0,0,0,0,0,0,0,0,0,0,0.5f,0,0.5f };

  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_SETTEXT:
    fgChild_Clear(&self->item);
    fgText_Create(msg->other, (void*)fgChild_VoidMessage((fgChild*)self, FG_GETFONT, 0), fgChild_VoidMessage((fgChild*)self, FG_GETFONTCOLOR, 0), FGCHILD_EXPANDX | FGCHILD_EXPANDY, &self->item, 0);
    return 0;
  case FG_ADDITEM:
    fgChild_Clear(&self->item);
    if(msg->other)
      fgChild_VoidMessage(&self->item, FG_ADDCHILD, msg->other);
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
  }
  return fgWindow_HoverProcess(&self->window,msg);
}