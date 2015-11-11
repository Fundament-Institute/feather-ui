// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgButton.h"
#include "fgSkin.h"

void FG_FASTCALL fgButton_Init(fgButton* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  assert(self!=0);
  memset(self,0,sizeof(fgButton));
  fgWindow_Init(&self->window,flags,parent,element);
  self->window.element.destroy = &fgButton_Destroy;
  self->window.element.message = &fgButton_Message;

  fgChild_Init(&self->item, FGCHILD_NOCLIP, (fgChild*)self, &fgElement_DEFAULT);
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
  case FG_SETCAPTION:
    fgChild_Clear(&self->item);
    fgChild_VoidMessage((fgChild*)self, FG_ADDCHILD, fgLoadDef(fgDefaultTextDef(0, msg->other), &defelem, 1));
    fgChild_TriggerStyle((fgChild*)self, 0);
    return 0;
  case FG_ADDITEM:
    fgChild_Clear(&self->item);
    fgChild_VoidMessage((fgChild*)self, FG_ADDCHILD, fgLoadDef(msg->other, &defelem, 1));
    fgChild_TriggerStyle((fgChild*)self, 0);
    return 0;
  case FG_NUETRAL:
    fgChild_TriggerStyle((fgChild*)self, 0);
    return 0;
  case FG_HOVER:
    fgChild_TriggerStyle((fgChild*)self, 1);
    return 0;
  case FG_ACTIVE:
    fgChild_TriggerStyle((fgChild*)self, 2);
    return 0;
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgButton";
    return 0;
  }
  return fgWindow_HoverProcess(&self->window,msg);
}