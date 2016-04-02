// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgMenu.h"
#include "feathercpp.h"

void FG_FASTCALL fgMenu_Init(fgMenu* self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, FG_UINT id, fgFlag flags, char submenu)
{
  assert(self!=0);
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgMenu_Destroy, (FN_MESSAGE)(submenu ? &fgSubmenu_Message : &fgMenu_Message));
}
void FG_FASTCALL fgMenu_Destroy(fgMenu* self)
{
  fgScrollbar_Destroy(&self->window); // this will destroy our prechildren for us.
  //fgRoot_DeallocAction(fgSingleton(),self->dropdown);
}

inline void FG_FASTCALL fgMenu_Show(fgMenu* self, bool show)
{
  fgFlag set = show ? (self->window.window.element.flags & ~(FGCHILD_HIDDEN | FGCHILD_IGNORE)) : (self->window.window.element.flags | FGCHILD_HIDDEN | FGCHILD_IGNORE);
  fgChild_IntMessage(*self, FG_SETFLAGS, set, 0);
}

size_t FG_FASTCALL fgMenu_Message(fgMenu* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgScrollbar_Message((fgScrollbar*)self, msg);
    fgChild_Init(&self->highlight, FGCHILD_HIDDEN | FGCHILD_IGNORE, *self, 0, &fgElement_DEFAULT);
    fgChild_AddPreChild(*self, &self->highlight);
    fgChild_Init(&self->arrow, FGCHILD_IGNORE | FGCHILD_EXPAND, 0, 0, 0);
    fgChild_AddPreChild(*self, &self->arrow);
    fgChild_AddPreChild(*self, 0); // seperator placeholder
    return 0;
  case FG_MOUSEDOWN:
  {
    char hit = MsgHitCRect(msg, *self);
    fgChild* child;
    AbsRect cache;
    if(hit)
      child = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(!hit || !child) // check if we are outside and need to close the menu
    {
      if(self->expanded)
        fgMenu_Show(self->expanded, false);
      self->expanded = 0;
      return 0;
    }
    size_t index = 0;
    if(((fgSubmenuArray&)self->submenus)[index]) // if this exists open the submenu
      fgMenu_Show(self->expanded = ((fgSubmenuArray&)self->submenus)[index], true);
    else // otherwise send an action message to this control (because what you clicked on may just be an image or text).
      fgSendMsg<FG_ACTION, void*>(*self, ((fgMenuArray&)self->members)[index]);
    return 0;
  }
    break;
  case FG_MOUSEUP:
    break;
  case FG_MOUSEMOVE:
  {
    AbsRect cache;
    fgChild* child = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child)
    {
      CRect r = { child->element.area.left.abs, 0, 0, 0, child->element.area.right.abs, 0, 0, 1 };
      fgSendMsg<FG_SETAREA, void*>(&self->highlight, &r);
    }
  }
    break;
  case FG_MOUSEOFF:
    if(!self->expanded) // Turn off the hover, but ONLY if a submenu isn't expanded.
      fgChild_IntMessage(&self->highlight, FG_SETFLAG, FGCHILD_HIDDEN, 1);
    break;
  case FG_ADDITEM:
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgMenu";
  }
  return fgScrollbar_Message((fgScrollbar*)self,msg);
}

size_t FG_FASTCALL fgSubmenu_Message(fgMenu* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN: // submenus respond to mouseup, however if the mousedown misses the control entirely we must propagate upwards
    return fgScrollbar_Message((fgScrollbar*)self, msg);
  case FG_MOUSEUP:
  {
    char hit = MsgHitCRect(msg, *self);
    fgChild* child;
    AbsRect cache;
    if(hit)
      child = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(!hit || !child) // check if we are outside and need to close the menu
    {
      fgMenu_Show(self, false);
      return 0;
    }
    size_t index = 0;
    if(((fgSubmenuArray&)self->submenus)[index]) // if this exists open the submenu
      fgMenu_Show(self->expanded = ((fgSubmenuArray&)self->submenus)[index], true);
    else // otherwise send an action message to ourselves (because what you clicked on may just be an image or text).
    {
      fgSendMsg<FG_ACTION, void*>(*self, ((fgMenuArray&)self->members)[index]);
      fgMenu_Show(self, false);
    }
    return 0;
  }
  case FG_MOUSEMOVE:
  {
    AbsRect cache;
    fgChild* child = fgChild_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child)
    {
      CRect r = { 0, 0, child->element.area.top.abs, 0, 0, 1, child->element.area.bottom.abs, 0 };
      fgSendMsg<FG_SETAREA, void*>(&self->highlight, &r);
    }
    return 0;
  }
  case FG_GETCLASSNAME:
    return (size_t)"fgSubmenu"; // This allows you to properly differentiate a top level menu glued to the top of a window from a submenu, like a context menu.
  }
  return fgMenu_Message(self, msg);
}