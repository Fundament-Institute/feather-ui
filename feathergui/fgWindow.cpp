// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgText.h"
#include "feathercpp.h"

size_t fgWindow_CloseMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    return _sendmsg<FG_ACTION, size_t>(self->control.element.parent, FGWINDOW_CLOSE); // This must immediately return because the control doesn't exist anymore
  return fgButton_Message(self, msg);
}

size_t fgWindow_MaximizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    _sendmsg<FG_ACTION, size_t>(self->control.element.parent, ((fgWindow*)self->control.element.parent)->maximized ? FGWINDOW_RESTORE : FGWINDOW_MAXIMIZE);
  return fgButton_Message(self, msg);
}

size_t fgWindow_MinimizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    _sendmsg<FG_ACTION, size_t>(self->control.element.parent, (self->control.element.parent->flags&FGELEMENT_HIDDEN) ? FGWINDOW_UNMINIMIZE : FGWINDOW_MINIMIZE);
  return fgButton_Message(self, msg);
}

void fgWindow_Init(fgWindow* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgWindow_Destroy, (fgMessage)&fgWindow_Message);
}
void fgWindow_Destroy(fgWindow* self)
{
  self->control->message = (fgMessage)fgControl_Message;
  fgControl_Destroy((fgControl*)self);
}

size_t fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    self->dragged = 0;
    self->maximized = 0;
    memset(&self->prevrect, 0, sizeof(CRect));
    fgText_Init(&self->caption, *self, 0, "Window$text", FGELEMENT_BACKGROUND | FGELEMENT_IGNORE | FGELEMENT_EXPAND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->controls[0], *self, 0, "Window$close", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->controls[1], *self, 0, "Window$restore", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->controls[2], *self, 0, "Window$minimize", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    self->controls[0].control.element.message = (fgMessage)&fgWindow_CloseMessage;
    self->controls[1].control.element.message = (fgMessage)&fgWindow_MaximizeMessage;
    self->controls[2].control.element.message = (fgMessage)&fgWindow_MinimizeMessage;
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgWindow* hold = reinterpret_cast<fgWindow*>(msg->e);
      self->caption->Clone(hold->caption);
      for(size_t i = 0; i < 3; ++i) self->controls[i]->Clone(hold->controls[i]);
      hold->dragged = 0;
      hold->prevrect = self->prevrect;
      hold->offset = self->offset;
      fgControl_Message(&self->control, msg);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->caption);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->controls[0]);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->controls[1]);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->controls[2]);
    }
    return sizeof(fgWindow);
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETLINEHEIGHT:
  case FG_SETLETTERSPACING:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETLINEHEIGHT:
  case FG_GETLETTERSPACING:
  case FG_GETCOLOR:
    return fgSendMessage(self->caption, msg);
  case FG_MOUSEDBLCLICK:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect(*self, &out);
      if(msg->y < out.top + self->control.element.padding.top)
        _sendmsg<FG_ACTION, size_t>(*self, FGWINDOW_MAXIMIZE);
      return FG_ACCEPT;
    }
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      self->offset.x = (FABS)msg->x;
      self->offset.y = (FABS)msg->y;

      AbsRect out;
      ResolveRect(*self, &out);
      AbsRect inner;
      GetInnerRect(*self, &inner, &out);
      AbsRect textout;
      ResolveRectCache(self->caption, &textout, &out, 0);

      if(!self->maximized) // Only resize/move if not maximized
      {
        // Check for resize
        if(self->control->flags&FGWINDOW_RESIZABLE)
        {
          if(msg->x < inner.left)
            self->dragged |= 2;
          if(msg->y < textout.top)
            self->dragged |= 4;
          if(msg->x >= inner.right)
            self->dragged |= 8;
          if(msg->y >= inner.bottom)
            self->dragged |= 16;
        }

        // Check for move
        if(msg->y >= textout.top &&
          msg->y < textout.bottom &&
          msg->x >= inner.left &&
          msg->x < inner.right)
          self->dragged = 1;
      }
      if(self->dragged != 0)
      {
        fgroot_instance->fgCaptureWindow = *self;
        _sendmsg<FG_ACTIVE>(*self);
      }
    }
    break;
  case FG_MOUSEMOVE:
    if(self->dragged & 1) // window is being moved around
    {
      AbsVec v = { (FABS)msg->x, (FABS)msg->y };
      CRect area = self->control.element.transform.area;
      MoveCRect(v.x - self->offset.x + area.left.abs, v.y - self->offset.y + area.top.abs, &area);
      self->offset = v;
      _sendmsg<FG_SETAREA, void*>(*self, &area);
    }
    else if(self->dragged)
    {
      AbsVec v = { (FABS)msg->x, (FABS)msg->y };
      CRect area = self->control.element.transform.area;
      if(self->dragged & 2) // resize on left
        area.left.abs += v.x - self->offset.x;
      if(self->dragged & 4) // resize on top
        area.top.abs += v.y - self->offset.y;
      if(self->dragged & 8) // resize on right
        area.right.abs += v.x - self->offset.x;
      if(self->dragged & 16) // resize on bottom 
        area.bottom.abs += v.y - self->offset.y;
      self->offset = v;
      _sendmsg<FG_SETAREA, void*>(*self, &area);
    } 

    if(fgControl_Message((fgControl*)self, msg))
    {
      AbsRect out;
      ResolveRect(*self, &out);
      AbsRect inner;
      GetInnerRect(*self, &inner, &out);
      AbsRect textout;
      ResolveRectCache(self->caption, &textout, &out, 0);

      size_t dragged = self->dragged;

      if(!dragged && (self->control->flags&FGWINDOW_RESIZABLE) != 0)
      {
      // Check for resize
        if(msg->x < inner.left)
          dragged |= 2;
        if(msg->y < textout.top)
          dragged |= 4;
        if(msg->x >= inner.right)
          dragged |= 8;
        if(msg->y >= inner.bottom)
          dragged |= 16;
      }

      if(memcmp(&self->control.element.transform, &fgTransform_DEFAULT, sizeof(fgTransform)) != 0) // Only resize/move if not maximized
      {
        if(dragged == 12 || dragged == 18)
          return FGCURSOR_RESIZENESW;
        if(dragged == 6 || dragged == 24)
          return FGCURSOR_RESIZENWSE;
        if(dragged == 2 || dragged == 8)
          return FGCURSOR_RESIZEWE;
        if(dragged == 4 || dragged == 16)
          return FGCURSOR_RESIZENS;
      }
      return FGCURSOR_ARROW;
    }
    return 0;
  case FG_MOUSEUP:
    self->dragged = 0;
    if(fgroot_instance->fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgroot_instance->fgCaptureWindow = 0;
    break;
  case FG_ACTION:
    switch(msg->i)
    {
    case FGWINDOW_MINIMIZE:
      _sendmsg<FG_SETFLAG, size_t, size_t>(*self, FGELEMENT_HIDDEN, 1);
      break;
    case FGWINDOW_UNMINIMIZE:
      _sendmsg<FG_SETFLAG, size_t, size_t>(*self, FGELEMENT_HIDDEN, 0);
      break;
    case FGWINDOW_MAXIMIZE:
      self->prevrect = self->control.element.transform.area;
      _sendmsg<FG_SETAREA, void*>(*self, (void*)&fgTransform_DEFAULT.area);
      self->maximized = 1;
      break;
    case FGWINDOW_RESTORE:
      _sendmsg<FG_SETAREA, void*>(*self, &self->prevrect);
      self->maximized = 0;
      break;
    case FGWINDOW_CLOSE:
      VirtualFreeChild(*self);
      return FG_ACCEPT;
    }
  case FG_INJECT:
    if(self->dragged != 0) // if we are being dragged, we completely bypass all children
      return (*fgroot_instance->backend.fgBehaviorHook)(*self, (const FG_Msg*)msg->p);
    break;
  case FG_GOTFOCUS:
    if(fgElement_CheckLastFocus(*self)) // try to resolve via lastfocus
      return FG_ACCEPT;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Window";
  }
  return fgControl_Message((fgControl*)self, msg);
}