// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgList.h"
#include "fgRoot.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

FG_EXTERN void FG_FASTCALL fgList_Init(fgList* self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgList_Destroy, (FN_MESSAGE)&fgList_Message);
}
FG_EXTERN void FG_FASTCALL fgList_Destroy(fgList* self)
{
  fgScrollbar_Destroy(&self->window); // this will destroy our prechildren for us.
}
FG_EXTERN size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->window.window.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message((fgWindow*)self, msg);
    self->selected = 0;
    fgChild_Init(&self->selector, FGCHILD_BACKGROUND, 0, 0, 0); // we do NOT set the parent of these because we need to manipulate when they get rendered.
    fgChild_Init(&self->hover, FGCHILD_BACKGROUND, 0, 0, 0);
    fgChild_AddPreChild(*self, &self->selector);
    fgChild_AddPreChild(*self, &self->hover);
    return 0;
  case FG_SETFLAG: // Do the same thing fgChild does to resolve a SETFLAG into SETFLAGS
    otherint = T_SETBIT(flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
    if((otherint^flags) & FGLIST_LAYOUTMASK)
    { // handle a layout flag change
      size_t r = fgScrollbar_Message(&self->window, msg); // we have to actually set the flags first before resetting the layout
      fgChild_SubMessage(*self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTRESET, 0, 0);
      return r;
    }
    break;
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEUP:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEMOVE:
    fgUpdateMouseState(&self->mouse, msg);
    if((self->window.window.element.flags&FGLIST_DRAGGABLE) && (self->mouse.state&FGMOUSE_DRAG)) // Check if we clicked inside this window
    {
      AbsRect cache;
      fgChild* target = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache); // find item below the mouse cursor (if any) and initiate a drag for it.
      if(target != 0)
        fgSendMsg<FG_DRAG, void*, const void*>(*self, target, msg);
    }
    break;
  case FG_DRAGGING:
    {
      fgChild* drag = fgroot_instance->drag;
      if(drag->parent != *self) // If something is being dragged over us, by default reject it if it wasn't from this list.
        break; // the default handler rejects it for us and sets the cursor.
      fgSetCursor(FGCURSOR_HAND, 0); // Set cursor to a droppable icon

      // TODO: need to schedule drawing a line on the next draw call here

    }
    return 0;
  case FG_DROP:
    if(msg->other)
    {
      fgChild* drag = (fgChild*)msg->other;
      if(drag->parent != *self)
        break; // drop to default handling to reject this if it isn't a child of this control

      AbsRect cache;
      AbsRect rect;
      fgChild* target = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      ResolveRectCache(target, &rect, &cache, (target->flags & FGCHILD_BACKGROUND) ? 0 : &target->padding);

      // figure out if we're on the x axis or y axis

      // Remove the child from where it currently is, then re-insert it above or below the control it's being dragged over.

      fgSetCursor(FGCURSOR_ARROW, 0);
      return 0;
    }
    return 0;
  case FG_LAYOUTFUNCTION:
    if(flags&(FGLIST_TILEX | FGLIST_TILEY)) // TILE flags override DISTRIBUTE flags, if they're specified.
      return fgLayout_Tile(*self, (const FG_Msg*)msg->other, (flags&FGLIST_LAYOUTMASK) >> 12);
    if(flags&(FGLIST_DISTRIBUTEX | FGLIST_DISTRIBUTEY))
      return fgLayout_Distribute(*self, (const FG_Msg*)msg->other, (flags&FGLIST_LAYOUTMASK) >> 12);
    break; // If no layout flags are specified, fall back to default layout behavior.
  }

  return fgScrollbar_Message(&self->window, msg);
}