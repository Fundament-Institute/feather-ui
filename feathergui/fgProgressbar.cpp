// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgProgressbar.h"
#include "fgSkin.h"

FG_EXTERN void FG_FASTCALL fgProgressbar_Init(fgProgressbar* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, element, (FN_DESTROY)&fgProgressbar_Destroy, (FN_MESSAGE)&fgProgressbar_Message);
}

FG_EXTERN void FG_FASTCALL fgProgressbar_Destroy(fgProgressbar* self)
{
  fgWindow_Destroy(&self->window);
}

FG_EXTERN size_t FG_FASTCALL fgProgressbar_Message(fgProgressbar* self, const FG_Msg* msg)
{
  static const fgElement BAR_ELEMENT = { 0,0,0,0,0,1.0,0,0,0,0,0,0,0 };

  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, &fgElement_CENTER);
    fgChild_IntMessage((fgChild*)&self->text, FG_SETORDER, 1, 0);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->text);
    fgChild_Init(&self->bar, FGCHILD_EXPAND | FGCHILD_IGNORE, (fgChild*)self, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->bar);
    self->value = 0.0f;
    return 0;
  case FG_ADDITEM:
    if(msg->other)
      fgChild_VoidMessage(&self->bar, FG_ADDCHILD, msg->other);
    else
      fgChild_Clear(&self->bar);
    return 0;
  case FG_SETSTATE:
    if(msg->otherf != self->value)
    {
      self->value = msg->otherf;
      CRect area = self->bar.element.area;
      area.right.rel = self->value;
      fgChild_VoidMessage(&self->bar, FG_SETAREA, &area);
    }
    return 0;
  case FG_GETSTATE:
    if(msg->other)
      *((FREL*)msg->other) = self->value;
    return 0;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage((fgChild*)&self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"fgProgressbar";
  }
  return fgWindow_Message(&self->window, msg);
}
