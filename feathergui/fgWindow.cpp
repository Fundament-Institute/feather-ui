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
    fgIntMessage(self->control.element.parent, FG_ACTION, !memcmp(&self->control.element.parent->transform, &fgTransform_DEFAULT, sizeof(fgTransform))? FGWINDOW_RESTORE : FGWINDOW_MAXIMIZE, 0);
  return fgButton_Message(self, msg);
}

size_t FG_FASTCALL fgWindow_MinimizeMessage(fgButton* self, const FG_Msg* msg)
{
  if(msg->type == FG_ACTION)
    fgIntMessage(self->control.element.parent, FG_ACTION, (self->control.element.parent->flags&FGELEMENT_HIDDEN)? FGWINDOW_UNMINIMIZE : FGWINDOW_MINIMIZE, 0);
  return fgButton_Message(self, msg);
}

void FG_FASTCALL fgWindow_Init(fgWindow* self, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, flags, 0, 0, transform, (FN_DESTROY)&fgWindow_Destroy, (FN_MESSAGE)&fgWindow_Message);
}
void FG_FASTCALL fgWindow_Destroy(fgWindow* self)
{  
  fgControl_Destroy((fgControl*)self);
}

size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message((fgControl*)self, msg);
    self->dragged = 0;
    fgText_Init(&self->caption, 0, 0, 0, FGELEMENT_BACKGROUND | FGELEMENT_IGNORE | FGELEMENT_EXPAND, *self, 0, 0);
    fgButton_Init(&self->controls[0], FGELEMENT_BACKGROUND, *self, 0, 0);
    fgButton_Init(&self->controls[1], FGELEMENT_BACKGROUND, *self, 0, 0);
    fgButton_Init(&self->controls[2], FGELEMENT_BACKGROUND, *self, 0, 0);
    fgElement_AddPreChild(*self, self->caption);
    fgElement_AddPreChild(*self, self->controls[0]);
    fgElement_AddPreChild(*self, self->controls[1]);
    fgElement_AddPreChild(*self, self->controls[2]);
    self->controls[0].control.element.message = (FN_MESSAGE)&fgWindow_CloseMessage;
    self->controls[1].control.element.message = (FN_MESSAGE)&fgWindow_MaximizeMessage;
    self->controls[2].control.element.message = (FN_MESSAGE)&fgWindow_MinimizeMessage;
    return 1;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETCOLOR:
    return fgPassMessage(self->caption, msg);
  case FG_MOUSEDBLCLICK:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect(*self, &out);
      if(msg->y < out.top + self->control.element.padding.top)
        fgIntMessage(*self, FG_ACTION, FGWINDOW_MAXIMIZE, 0);
      return 1;
    }
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      AbsRect out;
      ResolveRect(*self, &out);

      // Check for resize

      // Check for move
      if(msg->y < out.top + self->control.element.padding.top)
      {
        self->offset.x = msg->x;
        self->offset.y = msg->y;
        self->dragged = 1;
      }

      if(self->dragged)
      {
        fgCaptureWindow = *self;
        _sendmsg<FG_ACTIVE>(*self);
      }
    }
    break;
  case FG_MOUSEMOVE:
    if(self->dragged&1) // window is being moved around
    {
      AbsVec v = { msg->x, msg->y };
      CRect area = self->control.element.transform.area;
      AbsVec dv = { v.x - self->offset.x + area.left.abs, v.y - self->offset.y + area.top.abs };
      MoveCRect(dv, &area);
      self->offset = v;
      _sendmsg<FG_SETAREA, void*>(*self, &area);
      _sendmsg<FG_SETAREA, void*>(*self, &area);
    }
    if(self->dragged & 2) {} // resize on left
    if(self->dragged & 4) {} // resize on top
    if(self->dragged & 8) {} // resize on right
    if(self->dragged & 16) {} // resize on bottom
    break;
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
      return 1;
    }
  case FG_GETCLASSNAME:
    return (size_t)"fgWindow";
  }
  return fgControl_Message((fgControl*)self,msg);
}