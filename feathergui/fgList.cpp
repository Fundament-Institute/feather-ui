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
  fgBox_Destroy(&self->box); // this will destroy our prechildren for us.
}
FG_EXTERN size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->box.window.window.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Message(&self->box, msg);
    self->selected = 0;
    fgChild_Init(&self->selector, FGCHILD_BACKGROUND, 0, 0, 0); // we do NOT set the parent of these because we need to manipulate when they get rendered.
    fgChild_Init(&self->hover, FGCHILD_BACKGROUND, 0, 0, 0);
    fgChild_AddPreChild(*self, &self->selector);
    fgChild_AddPreChild(*self, &self->hover);
    return 1;
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEUP:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEMOVE:
    fgUpdateMouseState(&self->mouse, msg);
    if(!fgroot_instance->drag && (self->box.window.window.element.flags&FGLIST_DRAGGABLE) && (self->mouse.state&FGMOUSE_INSIDE)) // Check if we clicked inside this window
    {
      AbsRect cache;
      fgChild* target = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache); // find item below the mouse cursor (if any) and initiate a drag for it.
      if(target != 0)
        _sendmsg<FG_DRAG, void*, const void*>(*self, target, msg);
    }
    break;
  case FG_DRAGGING:
  {
    fgChild* drag = fgroot_instance->drag;
    if(drag->parent != *self) // If something is being dragged over us, by default reject it if it wasn't from this list.
      break; // the default handler rejects it for us and sets the cursor.
    fgSetCursor(FGCURSOR_HAND, 0); // Set cursor to a droppable icon
  }
  return 1;
  case FG_DROP:
    if(msg->other)
    {
      fgChild* drag = (fgChild*)msg->other;
      if(drag->parent != *self)
        break; // drop to default handling to reject this if it isn't a child of this control

      AbsRect cache;
      AbsRect rect;
      fgChild* target = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      ResolveRectCache(target, &rect, &cache, (target->flags & FGCHILD_BACKGROUND) ? 0 : &(*self)->padding);

      // figure out if we're on the x axis or y axis

      // Remove the child from where it currently is, then re-insert it above or below the control it's being dragged over.

      fgSetCursor(FGCURSOR_ARROW, 0);
    }
    return 1;
  case FG_DRAW:
    if(!(msg->subtype & 1))
    {
    // If dragging, draw a line, otherwise, draw the hover rect.


    // If there are selections, draw them here.
    }
    break;
  }

  return fgBox_Message(&self->box, msg);
}