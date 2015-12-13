// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTopWindow.h"
#include "fgText.h"

size_t FG_FASTCALL fgTopWindow_CloseMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgChild_IntMessage(self->window.element.parent, FG_ACTION, FGTOPWINDOW_CLOSE, 0);
  return fgButton_Message(self, msg);
}

size_t FG_FASTCALL fgTopWindow_MaximizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgChild_IntMessage(self->window.element.parent, FG_ACTION, !memcmp(&self->window.element.parent->element, &fgElement_DEFAULT, sizeof(fgElement))? FGTOPWINDOW_RESTORE : FGTOPWINDOW_MAXIMIZE, 0);
  return fgButton_Message(self, msg);
}

size_t FG_FASTCALL fgTopWindow_MinimizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgChild_IntMessage(self->window.element.parent, FG_ACTION, (self->window.element.parent->flags&FGCHILD_HIDDEN)? FGTOPWINDOW_UNMINIMIZE : FGTOPWINDOW_MINIMIZE, 0);
  return fgButton_Message(self, msg);
}

void FG_FASTCALL fgTopWindow_Init(fgTopWindow* self, fgFlag flags, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self, flags, 0, element, (FN_DESTROY)&fgTopWindow_Destroy, (FN_MESSAGE)&fgTopWindow_Message);
}
void FG_FASTCALL fgTopWindow_Destroy(fgTopWindow* self)
{  
  fgWindow_Destroy((fgWindow*)self);
}

size_t FG_FASTCALL fgTopWindow_Message(fgTopWindow* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message((fgWindow*)self, msg);
    self->dragged = 0;
    fgText_Init(&self->caption, 0, 0, 0, FGCHILD_BACKGROUND | FGCHILD_IGNORE | FGCHILD_EXPAND, (fgChild*)self, 0);
    fgButton_Init(&self->controls[0], FGCHILD_BACKGROUND, (fgChild*)self, 0);
    fgButton_Init(&self->controls[1], FGCHILD_BACKGROUND, (fgChild*)self, 0);
    fgButton_Init(&self->controls[2], FGCHILD_BACKGROUND, (fgChild*)self, 0);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->caption);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[0]);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[1]);
    fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[2]);
    self->controls[0].window.element.message = (FN_MESSAGE)&fgTopWindow_CloseMessage;
    self->controls[1].window.element.message = (FN_MESSAGE)&fgTopWindow_MaximizeMessage;
    self->controls[2].window.element.message = (FN_MESSAGE)&fgTopWindow_MinimizeMessage;
    return 0;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgChild_PassMessage((fgChild*)&self->caption, msg);
  case FG_MOUSEDBLCLICK:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect((fgChild*)self, &out);
      if(msg->y < out.top + self->window.element.padding.top)
        fgChild_IntMessage((fgChild*)self, FG_ACTION, FGTOPWINDOW_MAXIMIZE, 0);
      return 0;
    }
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect((fgChild*)self, &out);

      // Check for resize

      // Check for move
      if(msg->y < out.top + self->window.element.padding.top)
      {
        self->offset.x = msg->x;
        self->offset.y = msg->y;
        self->dragged = 1;
      }

      if(self->dragged)
      {
        fgCaptureWindow = (fgChild*)self;
        fgChild_VoidMessage((fgChild*)self, FG_ACTIVE, 0);
      }
    }
    break;
  case FG_MOUSEMOVE:
    if(self->dragged&1) // window is being moved around
    {
      AbsVec v = { msg->x, msg->y };
      CRect area = self->window.element.element.area;
      AbsVec dv = { v.x - self->offset.x + area.left.abs, v.y - self->offset.y + area.top.abs };
      MoveCRect(dv, &area);
      self->offset = v;
      fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &area);
    }
    if(self->dragged & 2) {} // resize on left
    if(self->dragged & 4) {} // resize on top
    if(self->dragged & 8) {} // resize on right
    if(self->dragged & 16) {} // resize on bottom
    break;
  case FG_MOUSEUP:
    self->dragged = 0;
    if(fgCaptureWindow == (fgChild*)self) // Remove our control hold on mouse messages.
      fgCaptureWindow = 0;
    break;
  case FG_ACTION:
    switch(msg->otherint)
    {
    case FGTOPWINDOW_MINIMIZE:
      fgChild_IntMessage((fgChild*)self, FG_SETFLAG, FGCHILD_HIDDEN, 1);
      break;
    case FGTOPWINDOW_UNMINIMIZE:
      fgChild_IntMessage((fgChild*)self, FG_SETFLAG, FGCHILD_HIDDEN, 0);
      break;
    case FGTOPWINDOW_MAXIMIZE:
      self->prevrect = self->window.element.element.area;
      fgChild_VoidMessage((fgChild*)self, FG_SETAREA, (void*)&fgElement_DEFAULT.area);
      break;
    case FGTOPWINDOW_RESTORE:
      fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &self->prevrect);
      break;
    case FGTOPWINDOW_CLOSE:
      VirtualFreeChild((fgChild*)self);
      return 0;
    }
  case FG_GETCLASSNAME:
    return (size_t)"fgTopWindow";
  }
  return fgWindow_Message((fgWindow*)self,msg);
}