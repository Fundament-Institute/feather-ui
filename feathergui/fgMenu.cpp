// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgMenu.h"
#include "feathercpp.h"

void FG_FASTCALL fgMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT prev, const fgTransform* transform, FG_UINT id, fgFlag flags, char submenu)
{
  assert(self != 0);
  fgElement_InternalSetup(*self, flags, parent, prev, transform, (FN_DESTROY)&fgMenu_Destroy, (FN_MESSAGE)(submenu ? &fgSubmenu_Message : &fgMenu_Message));
}
void FG_FASTCALL fgMenu_Destroy(fgMenu* self)
{
  fgScrollbar_Destroy(&self->window); // this will destroy our prechildren for us.
                                      //fgRoot_DeallocAction(fgSingleton(),self->dropdown);
}

inline void FG_FASTCALL fgMenu_Show(fgMenu* self, bool show)
{
  fgFlag set = show ? (self->window.control.element.flags & ~(FGELEMENT_HIDDEN | FGELEMENT_IGNORE)) : (self->window.control.element.flags | FGELEMENT_HIDDEN | FGELEMENT_IGNORE);
  fgIntMessage(*self, FG_SETFLAGS, set, 0);
}

size_t FG_FASTCALL fgMenu_Message(fgMenu* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgScrollbar_Message((fgScrollbar*)self, msg);
    fgElement_Init(&self->highlight, FGELEMENT_HIDDEN | FGELEMENT_IGNORE, *self, 0, &fgTransform_DEFAULT);
    fgElement_AddPreChild(*self, &self->highlight);
    fgElement_Init(&self->arrow, FGELEMENT_IGNORE | FGELEMENT_EXPAND, 0, 0, 0);
    fgElement_AddPreChild(*self, &self->arrow);
    fgElement_AddPreChild(*self, 0); // seperator placeholder
    return 1;
  case FG_MOUSEDOWN:
  {
    if(self->expanded) // for top level menus, if there is an expanded submenu and it did not handle a mousedown, we always close the menu.
    {
      fgMenu_Show(self->expanded, false);
      self->expanded = 0;
      if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
        fgCaptureWindow = 0;
      return 1;
    }

    assert(fgCaptureWindow != *self); // this should never happen (if it does you may need to always remove the capture window status).
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child) // If you click the empty part of the menu, nothing happens, but if you hit a child, we check if it has a submenu
    {
      fgMenu* submenu = reinterpret_cast<fgMenu*>(_sendmsg<FG_GETSELECTEDITEM>(child));
      if(submenu) // if this exists open the submenu
      {
        fgCaptureWindow = *self; // Because we are the top level menu, we must also capture the mouse
        fgMenu_Show(self->expanded = submenu, true);
      }
      else // otherwise send an action message to ourselves (because what you clicked on may just be an image or text).
        _sendmsg<FG_ACTION, void*>(*self, child);
    }

    return 1;
  }
  break;
  case FG_MOUSEUP:
    break; // TODO: You may need to block all mouse messages from propagating down and manually trigger the hover calculations, because otherwise the mouse capture will break it.
  case FG_MOUSEMOVE:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child)
    {
      CRect r = { child->transform.area.left.abs, 0, 0, 0, child->transform.area.right.abs, 0, 0, 1 };
      _sendmsg<FG_SETAREA, void*>(&self->highlight, &r);
    }
  }
  break;
  case FG_MOUSEOFF:
    break; // the top level menu never turns off its hover
  case FG_ADDITEM:
    return 1;
  case FG_GETCLASSNAME:
    return (size_t)"fgMenu";
  }
  return fgScrollbar_Message((fgScrollbar*)self, msg);
}

size_t FG_FASTCALL fgSubmenu_Message(fgMenu* self, const FG_Msg* msg)
{
  /*switch(msg->type)
  {
  case FG_MOUSEDOWN: // submenus respond to mouseup, however if the mousedown misses the control entirely we must propagate upwards
  return fgScrollbar_Message((fgScrollbar*)self, msg);
  case FG_MOUSEUP:
  {
  char hit = MsgHitCRect(msg, *self);
  fgElement* child;
  AbsRect cache;
  if(hit)
  child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
  if(!hit || !child) // check if we are outside and need to close the menu
  {
  fgMenu_Show(self, false);
  return 1;
  }
  size_t index = 0;
  if(((fgSubmenuArray&)self->submenus)[index]) // if this exists open the submenu
  fgMenu_Show(self->expanded = ((fgSubmenuArray&)self->submenus)[index], true);
  else // otherwise send an action message to ourselves (because what you clicked on may just be an image or text).
  {
  _sendmsg<FG_ACTION, void*>(*self, ((fgMenuArray&)self->members)[index]);
  fgMenu_Show(self, false);
  }
  return 1;
  }
  case FG_MOUSEMOVE:
  {
  AbsRect cache;
  fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
  if(child)
  {
  CRect r = { 0, 0, child->transform.area.top.abs, 0, 0, 1, child->transform.area.bottom.abs, 0 };
  _sendmsg<FG_SETAREA, void*>(&self->highlight, &r);
  }
  return 1;
  }
  case FG_MOUSEOFF:
  if(!self->expanded) // Turn off the hover, but ONLY if a submenu isn't expanded.
  fgIntMessage(&self->highlight, FG_SETFLAG, FGELEMENT_HIDDEN, 1);
  case FG_GETCLASSNAME:
  return (size_t)"fgSubmenu"; // This allows you to properly differentiate a top level menu glued to the top of a window from a submenu, like a context menu.
  }*/
  return fgScrollbar_Message(&self->window, msg);
}