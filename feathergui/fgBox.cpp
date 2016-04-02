// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBox.h"
#include "fgRoot.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgBox_Init(fgBox* self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgBox_Destroy, (FN_MESSAGE)&fgBox_Message);
}
void FG_FASTCALL fgBox_Destroy(fgBox* self)
{
  fgScrollbar_Destroy(&self->window); // this will destroy our prechildren for us.
}
size_t FG_FASTCALL fgBox_Message(fgBox* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->window.window.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    break; // No constructor
  case FG_SETFLAG: // Do the same thing fgChild does to resolve a SETFLAG into SETFLAGS
    otherint = T_SETBIT(flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
    if((otherint^flags) & FGBOX_LAYOUTMASK)
    { // handle a layout flag change
      size_t r = fgScrollbar_Message(&self->window, msg); // we have to actually set the flags first before resetting the layout
      fgChild_SubMessage(*self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTRESET, 0, 0);
      return r;
    }
    break;
  case FG_LAYOUTFUNCTION:
    if(flags&(FGBOX_TILEX | FGBOX_TILEY)) // TILE flags override DISTRIBUTE flags, if they're specified.
      return fgLayout_Tile(*self, (const FG_Msg*)msg->other, (flags&FGBOX_LAYOUTMASK) >> 12);
    if(flags&(FGBOX_DISTRIBUTEX | FGBOX_DISTRIBUTEY))
      return fgLayout_Distribute(*self, (const FG_Msg*)msg->other, (flags&FGBOX_LAYOUTMASK) >> 12);
    break; // If no layout flags are specified, fall back to default layout behavior.
  }

  return fgScrollbar_Message(&self->window, msg);
}