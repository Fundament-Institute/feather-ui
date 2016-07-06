// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgList.h"
#include "fgRoot.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgList_Destroy, (fgMessage)&fgList_Message);
}
void FG_FASTCALL fgList_Destroy(fgList* self)
{
  fgBox_Destroy(&self->box); // this will destroy our prechildren for us.
}
size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->box.window.control.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    memset(&self->selected, 0, sizeof(fgVectorElement));
    memset(&self->mouse, 0, sizeof(fgMouseState));
    self->select.color = 0x99999999;
    self->hover.color = 0x99999999;
    break;
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEUP:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEMOVE:
    fgUpdateMouseState(&self->mouse, msg);
    if(!fgroot_instance->drag && (self->box.window.control.element.flags&FGLIST_DRAGGABLE) && (self->mouse.state&FGMOUSE_INSIDE)) // Check if we clicked inside this window
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache); // find item below the mouse cursor (if any) and initiate a drag for it.
      if(target != 0)
        _sendmsg<FG_DRAG, void*, const void*>(*self, target, msg);
    }
    break;
  case FG_DRAGGING:
  {
    fgElement* drag = fgroot_instance->drag;
    if(drag->parent != *self) // If something is being dragged over us, by default reject it if it wasn't from this list.
      break; // the default handler rejects it for us and sets the cursor.
    fgSetCursor(FGCURSOR_HAND, 0); // Set cursor to a droppable icon
  }
    return FG_ACCEPT;
  case FG_DROP:
    if(msg->other)
    {
      fgElement* drag = (fgElement*)msg->other;
      if(drag->parent != *self)
        break; // drop to default handling to reject this if it isn't a child of this control

      AbsRect cache;
      AbsRect rect;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      ResolveRectCache(target, &rect, &cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &(*self)->padding);

      // figure out if we're on the x axis or y axis

      // Remove the child from where it currently is, then re-insert it above or below the control it's being dragged over.

      fgSetCursor(FGCURSOR_ARROW, 0);
    }
    return FG_ACCEPT;
  case FG_DRAW:
    if(!(msg->subtype & 1))
    {
      if(self->mouse.state&FGMOUSE_DRAG)
      {
        //draw line
      }
    }
    break;
  case FG_GETCLASSNAME:
    return (size_t)"List";
  }

  return fgBox_Message(&self->box, msg);
}