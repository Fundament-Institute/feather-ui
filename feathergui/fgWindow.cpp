// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgText.h"
#include "feathercpp.h"

size_t FG_FASTCALL fgWindow_CloseMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgIntMessage(self->control.element.parent, FG_ACTION, FGWINDOW_CLOSE, 0);
  return fgButton_Message(self, msg);
}

size_t FG_FASTCALL fgWindow_MaximizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgIntMessage(self->control.element.parent, FG_ACTION, !memcmp(&self->control.element.parent->transform, &fgTransform_DEFAULT, sizeof(fgTransform)) ? FGWINDOW_RESTORE : FGWINDOW_MAXIMIZE, 0);
  return fgButton_Message(self, msg);
}

size_t FG_FASTCALL fgWindow_MinimizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgIntMessage(self->control.element.parent, FG_ACTION, (self->control.element.parent->flags&FGELEMENT_HIDDEN) ? FGWINDOW_UNMINIMIZE : FGWINDOW_MINIMIZE, 0);
  return fgButton_Message(self, msg);
}

void FG_FASTCALL fgWindow_Init(fgWindow* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgWindow_Destroy, (fgMessage)&fgWindow_Message);
}
void FG_FASTCALL fgWindow_Destroy(fgWindow* self)
{
  fgControl_Destroy((fgControl*)self);
}

size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message((fgControl*)self, msg);
    self->dragged = 0;
    fgText_Init(&self->caption, *self, 0, "Window$text", FGELEMENT_BACKGROUND | FGELEMENT_IGNORE | FGELEMENT_EXPAND, 0, 0);
    fgButton_Init(&self->controls[0], *self, 0, "Window$close", FGELEMENT_BACKGROUND, 0, 0);
    fgButton_Init(&self->controls[1], *self, 0, "Window$restore", FGELEMENT_BACKGROUND, 0, 0);
    fgButton_Init(&self->controls[2], *self, 0, "Window$minimize", FGELEMENT_BACKGROUND, 0, 0);
    self->controls[0].control.element.message = (fgMessage)&fgWindow_CloseMessage;
    self->controls[1].control.element.message = (fgMessage)&fgWindow_MaximizeMessage;
    self->controls[2].control.element.message = (fgMessage)&fgWindow_MinimizeMessage;
    return FG_ACCEPT;
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
    return fgPassMessage(self->caption, msg);
  case FG_MOUSEDBLCLICK:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect(*self, &out);
      if(msg->y < out.top + self->control.element.padding.top)
        fgIntMessage(*self, FG_ACTION, FGWINDOW_MAXIMIZE, 0);
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

      if(memcmp(&self->control.element.transform, &fgTransform_DEFAULT, sizeof(fgTransform)) != 0) // Only resize/move if not maximized
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
        fgCaptureWindow = *self;
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
          fgRoot_SetCursor(FGCURSOR_RESIZENESW, 0);
        if(dragged == 6 || dragged == 24)
          fgRoot_SetCursor(FGCURSOR_RESIZENWSE, 0);
        if(dragged == 2 || dragged == 8)
          fgRoot_SetCursor(FGCURSOR_RESIZEWE, 0);
        if(dragged == 4 || dragged == 16)
          fgRoot_SetCursor(FGCURSOR_RESIZENS, 0);
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_MOUSEUP:
    self->dragged = 0;
    if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgCaptureWindow = 0;
    break;
  case FG_ACTION:
    switch(msg->otherint)
    {
    case FGWINDOW_MINIMIZE:
      fgIntMessage(*self, FG_SETFLAG, FGELEMENT_HIDDEN, 1);
      break;
    case FGWINDOW_UNMINIMIZE:
      fgIntMessage(*self, FG_SETFLAG, FGELEMENT_HIDDEN, 0);
      break;
    case FGWINDOW_MAXIMIZE:
      self->prevrect = self->control.element.transform.area;
      _sendmsg<FG_SETAREA, void*>(*self, (void*)&fgTransform_DEFAULT.area);
      break;
    case FGWINDOW_RESTORE:
      _sendmsg<FG_SETAREA, void*>(*self, &self->prevrect);
      break;
    case FGWINDOW_CLOSE:
      VirtualFreeChild(*self);
      return FG_ACCEPT;
    }
  case FG_INJECT:
    if(self->dragged != 0) // if we are being dragged, we completely bypass all children
      return (*fgroot_instance->backend.behaviorhook)(*self, (const FG_Msg*)msg->other);
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