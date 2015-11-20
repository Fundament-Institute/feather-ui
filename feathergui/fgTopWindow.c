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
  assert(self!=0);
  fgWindow_Init((fgWindow*)self, flags, 0, element);
  self->window.element.destroy = &fgTopWindow_Destroy;
  self->window.element.message = &fgTopWindow_Message;

  fgChild_Init(&self->region, 0, (fgChild*)self, &fgElement_DEFAULT);
  fgChild_Init(&self->titlebar, FGCHILD_BACKGROUND, (fgChild*)self, 0);
  fgButton_Init(&self->controls[0], FGCHILD_BACKGROUND, (fgChild*)self, 0);
  fgButton_Init(&self->controls[1], FGCHILD_BACKGROUND, (fgChild*)self, 0);
  fgButton_Init(&self->controls[2], FGCHILD_BACKGROUND, (fgChild*)self, 0);
  fgChild_AddPreChild((fgChild*)self, &self->region);
  fgChild_AddPreChild((fgChild*)self, &self->titlebar);
  fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[0]);
  fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[1]);
  fgChild_AddPreChild((fgChild*)self, (fgChild*)&self->controls[2]);
  self->controls[0].window.element.message = &fgTopWindow_CloseMessage;
  self->controls[1].window.element.message = &fgTopWindow_MaximizeMessage;
  self->controls[2].window.element.message = &fgTopWindow_MinimizeMessage;
}
void FG_FASTCALL fgTopWindow_Destroy(fgTopWindow* self)
{  
  assert(self!=0);
  fgChild_Destroy(&self->region);
  fgChild_Destroy(&self->titlebar);
  fgButton_Destroy(&self->controls[0]);
  fgButton_Destroy(&self->controls[1]);
  fgButton_Destroy(&self->controls[2]);
  fgWindow_Destroy((fgWindow*)self);
}

size_t FG_FASTCALL fgTopWindow_Message(fgTopWindow* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_REMOVECHILD:
    if(((fgChild*)msg->other)->parent==(fgChild*)self)
      return fgWindow_Message(&self->window,msg);
    return fgChild_Message(&self->region,msg);
  case FG_ADDCHILD:
    if(!(((fgChild*)msg->other)->flags&FGCHILD_BACKGROUND))
      return fgWindow_Message(&self->window,msg);
    return fgChild_Message(&self->region,msg);
  case FG_SETTEXT:
    fgChild_Clear(&self->titlebar);
    fgText_Create(msg->other, (void*)fgChild_VoidMessage((fgChild*)self, FG_GETFONT, 0), fgChild_VoidMessage((fgChild*)self, FG_GETFONTCOLOR, 0), FGCHILD_EXPANDX|FGCHILD_EXPANDY, (fgChild*)self, 0);
    fgChild_TriggerStyle((fgChild*)self, 0);
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