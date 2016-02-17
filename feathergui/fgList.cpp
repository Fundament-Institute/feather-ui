// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgList.h"
#include "bss-util\bss_util.h"

FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, FG_UINT id, fgFlag flags)
{

}
FG_EXTERN void FG_FASTCALL fgList_Destroy(fgList* self)
{

}
FG_EXTERN size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->window.window.element.flags;

  switch(msg->type)
  {
  case FG_SETFLAG: // Do the same thing fgChild does to resolve a SETFLAG into SETFLAGS
    otherint = T_SETBIT(flags, otherint, msg->otherintaux);
  case FG_SETFLAGS:
    if((otherint^flags) & FGLIST_LAYOUTMASK)
    { // handle a layout flag change
      size_t r = fgScrollbar_Message(&self->window, msg); // we have to actually set the flags first before resetting the layout
      fgChild_SubMessage((fgChild*)self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTRESET, 0, 0);
      return r;
    }
    break;
  case FG_LAYOUTFUNCTION:
    if(flags&(FGLIST_TILEX | FGLIST_TILEY)) // TILE flags override DISTRIBUTE flags, if they're specified.
      return fgLayout_Tile((fgChild*)self, (const FG_Msg*)msg->other1, (flags&FGLIST_LAYOUTMASK) >> 12);
    if(flags&(FGLIST_DISTRIBUTEX | FGLIST_DISTRIBUTEY))
      return fgLayout_Distribute((fgChild*)self, (const FG_Msg*)msg->other1, (flags&FGLIST_LAYOUTMASK) >> 12);
    break; // If no layout flags are specified, fall back to default layout behavior.
  }

  return fgScrollbar_Message(&self->window, msg);
}