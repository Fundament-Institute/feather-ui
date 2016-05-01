// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTreeView.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

FG_EXTERN void FG_FASTCALL fgTreeView_Init(fgTreeView* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{

}
FG_EXTERN void FG_FASTCALL fgTreeView_Destroy(fgTreeView* self)
{

}
FG_EXTERN size_t FG_FASTCALL fgTreeView_Message(fgTreeView* self, const FG_Msg* msg)
{
  fgElement* hold = (fgElement*)msg->other;

  switch(msg->type)
  {
  case FG_MOUSEDOWN:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSEMOVE:
  case FG_MOUSESCROLL:
  case FG_MOUSELEAVE:

    break;
  case FG_ADDITEM:
    return _sendmsg<FG_SETPARENT, void*>(hold, self);
  }
  return fgScrollbar_Message(&self->window, msg);
}