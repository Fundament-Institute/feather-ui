// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgList.h"
#include "fgRoot.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

static const char* FGSTR_LISTITEM = "ListItem";

void FG_FASTCALL fgListItem_Init(fgControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgElement_Destroy, (fgMessage)&fgListItem_Message);
}

size_t FG_FASTCALL fgListItem_Message(fgControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
  case FG_MOUSEMOVE:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
  case FG_MOUSELEAVE: // We process the mouse messages and then reject them anyway, which allows our parent to respond correctly.
    fgControl_HoverMessage(self, msg);
    return 0;
  case FG_NUETRAL:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "nuetral", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_HOVER:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "hover", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_ACTIVE:
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "active", fgStyleGetMask("nuetral", "hover", "active"));
    return FG_ACCEPT;
  case FG_GETCLASSNAME:
    return (size_t)FGSTR_LISTITEM;
  }

  return fgControl_HoverMessage(self, msg);
}

void FG_FASTCALL fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgList_Destroy, (fgMessage)&fgList_Message);
}
void FG_FASTCALL fgList_Destroy(fgList* self)
{
  ((bss_util::cArraySort<fgElement*>&)self->selected).~cArraySort();
  fgBox_Destroy(&self->box);
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
    self->select.color = 0xFF9999DD;
    self->hover.color = 0x99999999;
    break;
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    if(self->box->flags&FGLIST_SELECT)
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!target)
        break;
      size_t index = ((bss_util::cArraySort<fgElement*>&)self->selected).Find(target);
      if((self->box->flags&FGLIST_MULTISELECT) != FGLIST_MULTISELECT || !fgroot_instance->GetKey(FG_KEY_SHIFT))
      {
        for(size_t i = 0; i < self->selected.l; ++i)
          if(self->selected.p[i]->GetClassName() == FGSTR_LISTITEM)
            _sendsubmsg<FG_SETSTYLE, void*, size_t>(self->selected.p[i], 1, 0, fgStyleGetMask("selected"));
        ((bss_util::cArraySort<fgElement*>&)self->selected).Clear();
      }
      else if(index != (size_t)-1)
      {
        if(self->selected.p[index]->GetClassName() == FGSTR_LISTITEM)
          _sendsubmsg<FG_SETSTYLE, void*, size_t>(self->selected.p[index], 1, 0, fgStyleGetMask("selected"));
        ((bss_util::cArraySort<fgElement*>&)self->selected).Remove(index);
      }

      if(index == (size_t)-1)
      {
        if(target->GetClassName() == FGSTR_LISTITEM)
          _sendsubmsg<FG_SETSTYLE, void*, size_t>(target, 0, "selected", fgStyleGetMask("selected"));
        ((bss_util::cArraySort<fgElement*>&)self->selected).Insert(target);
      }
    }
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
      if(!target)
        break;
      ResolveRectCache(target, &rect, &cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &(*self)->padding);

      // figure out if we're on the x axis or y axis

      // Remove the child from where it currently is, then re-insert it above or below the control it's being dragged over.

      fgSetCursor(FGCURSOR_ARROW, 0);
    }
    return FG_ACCEPT;
  case FG_DRAW:
    if(!(msg->subtype & 1))
    {
      for(size_t i = 0; i < self->selected.l; ++i)
      {
        if(self->selected.p[i]->GetClassName() != FGSTR_LISTITEM)
        {
          AbsRect r;
          ResolveRectCache(self->selected.p[i], &r, (AbsRect*)msg->other, (self->selected.p[i]->flags & FGELEMENT_BACKGROUND) ? 0 : &(*self)->padding);
          fgDrawResource(0, &CRect { 0 }, self->select.color, 0, 0.0f, &r, 0.0f, &AbsVec { 0,0 }, FGRESOURCE_ROUNDRECT);
        }
      }

      if(self->mouse.state&FGMOUSE_DRAG)
      {
        //draw line
      }
      else
      {
        AbsRect cache;
        fgElement* target = fgElement_GetChildUnderMouse(*self, self->mouse.x, self->mouse.y, &cache);
        if(target && target->GetClassName() != FGSTR_LISTITEM)
        {
          AbsRect r;
          ResolveRectCache(target, &r, &cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &(*self)->padding);
          fgDrawResource(0, &CRect { 0 }, self->hover.color, 0, 0.0f, &r, 0.0f, &AbsVec { 0,0 }, FGRESOURCE_ROUNDRECT);
        }
      }
    }
    break;
  case FG_GETCOLOR:
    return !msg->otherint ? self->select.color : self->hover.color;
  case FG_SETCOLOR:
    if(!msg->otheraux)
      self->select.color = (size_t)msg->otherint;
    else
      self->hover.color = (size_t)msg->otherint;
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return ((size_t)msg->otherint) < self->selected.l ? (size_t)self->selected.p[(size_t)msg->otherint] : 0;
  case FG_GETCLASSNAME:
    return (size_t)"List";
  }

  return fgBox_Message(&self->box, msg);
}