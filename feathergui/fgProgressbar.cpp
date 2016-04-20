// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgProgressbar.h"
#include "fgSkin.h"
#include "feathercpp.h"

FG_EXTERN void FG_FASTCALL fgProgressbar_Init(fgProgressbar* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgProgressbar_Destroy, (FN_MESSAGE)&fgProgressbar_Message);
}

FG_EXTERN void FG_FASTCALL fgProgressbar_Destroy(fgProgressbar* self)
{
  fgWindow_Destroy(&self->window);
}

FG_EXTERN size_t FG_FASTCALL fgProgressbar_Message(fgProgressbar* self, const FG_Msg* msg)
{
  static const fgElement BAR_ELEMENT = { 0,0,0,0,0,0,0,1.0,0,0,0,0,0 };

  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    fgChild_Init(&self->bar, FGCHILD_BACKGROUND | FGCHILD_IGNORE, *self, 0, &BAR_ELEMENT);
    fgChild_AddPreChild(*self, &self->bar);
    fgText_Init(&self->text, 0, 0, 0, FGCHILD_EXPAND | FGCHILD_IGNORE, *self, 0, &fgElement_CENTER);
    fgChild_AddPreChild(*self, self->text);
    self->value = 0.0f;
    return 1;
  case FG_ADDITEM:
    if(msg->other)
      _sendmsg<FG_ADDCHILD, void*>(&self->bar, msg->other);
    else
      fgChild_Clear(&self->bar);
    return 1;
  case FG_SETSTATE:
    if(msg->otherf != self->value)
    {
      self->value = msg->otherf;
      CRect area = self->bar.element.area;
      area.right.rel = self->value;
      _sendmsg<FG_SETAREA, void*>(&self->bar, &area);
    }
    return 1;
  case FG_GETSTATE:
    return *reinterpret_cast<size_t*>(&self->value);
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage(self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"fgProgressbar";
  }
  return fgWindow_Message(&self->window, msg);
}
