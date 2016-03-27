// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgRoot.h"
#include "feathercpp.h"

fgChild* fgFocusedWindow = 0;
fgChild* fgLastHover = 0;
fgChild* fgCaptureWindow = 0;

void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(&self->element, flags, parent, prev, element, (FN_DESTROY)&fgWindow_Destroy, (FN_MESSAGE)&fgWindow_Message);
}

void FG_FASTCALL fgWindow_Destroy(fgWindow* self)
{
  assert(self!=0);
  if(self->tabprev) self->tabprev->tabnext = self->tabnext;
  if(self->tabnext) self->tabnext->tabprev = self->tabprev;
  self->tabnext = self->tabprev = 0;

  if(self->name) free(self->name);
  fgChild_Destroy(&self->element);
}

void FG_FASTCALL fgWindow_DoHoverCalc(fgWindow* self) // This is a seperate function so some controls can force a focus event to instead activate hover instead.
{
  if(fgLastHover != *self)
  {
    if(fgLastHover != 0)
      fgSendMsg<FG_MOUSEOFF>(fgLastHover);
    fgLastHover = *self;
    fgSendMsg<FG_MOUSEON>(*self);
  }
}

size_t FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  assert(self!=0);
  assert(msg!=0);
  
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgChild_Message(*self, msg);
    self->contextmenu = 0;
    self->name = 0;
    self->tabnext = self->tabprev = self; // This creates an infinite loop of tabbing
    return 0;
  case FG_KEYDOWN:
    if(msg->keycode == FG_KEY_TAB && !(msg->sigkeys&(2|4)))
    {
      fgWindow* target = (msg->sigkeys & 1) ? self->tabprev : self->tabnext;
      if(target != self)
        fgSendMsg<FG_GOTFOCUS>(*self);
      return 0;
    }
    break;
  case FG_MOUSEDOWN:
    fgWindow_DoHoverCalc(self);
    if(fgFocusedWindow != *self)
      fgSendMsg<FG_GOTFOCUS>(*self);
    //if(msg->button == FG_MOUSERBUTTON && self->contextmenu != 0)
    //  fgSendMsg<FG_GOTFOCUS>(*self->contextmenu);
    return 0;
  case FG_MOUSESCROLL:
    fgWindow_DoHoverCalc(self);
    return 0;
  case FG_MOUSEMOVE:
    fgWindow_DoHoverCalc(self);
    if(fgroot_instance->drag) // Send a dragging message if necessary. Does not initiate a drag for you (this is because some drags are initiated via click and drag, and some are just clicking).
    {
      FG_Msg m = *msg;
      m.type = FG_DRAGGING;
      fgChild_PassMessage(*self, &m);
    }
    return 0;
  case FG_MOUSEUP: 
  { // Any control that gets a MOUSEUP event immediately fires a MOUSEMOVE event at that location, which will force the focus to shift to a different control if the mouseup occured elsewhere.
    FG_Msg m = *msg;
    m.type = FG_MOUSEMOVE;
    fgRoot_Inject(fgroot_instance, &m);
  }
    if(fgroot_instance->drag != 0) // If necessary, send a drop message to the current control we're hovering over.
    {
      fgSendMsg<FG_DROP, void*>(fgLastHover, fgroot_instance->drag);
      fgroot_instance->drag = 0; // Ensure the drag pointer is set to NULL even if the target window doesn't understand the FG_DROP message.
    }
    return 0;
  case FG_GOTFOCUS:
    if(!fgChild_Message(*self, msg)) // checks if we resolved via lastfocus.
      return 0;
    if(fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      fgSendMsg<FG_LOSTFOCUS, void*>(fgFocusedWindow, self);
    fgFocusedWindow = *self;
    return 0;
  case FG_LOSTFOCUS:
    assert(fgFocusedWindow == *self);
    if(fgFocusedWindow == *self)
    {
      fgFocusedWindow = 0;
      if(self->element.parent)
      {
        if(!msg->other)
          fgSendMsg<FG_GOTFOCUS>(self->element.parent);
        else
          self->element.parent->lastfocus = *self;
      }
    }
    return 0;
  case FG_CLONE:
  {
    fgWindow* hold = (fgWindow*)msg->other;
    if(!hold)
      hold = (fgWindow*)malloc(sizeof(fgWindow));
    hold->contextmenu = hold->contextmenu;
    fgSendMsg<FG_SETNAME, void*>(*hold, self->name);

    FG_Msg m = *msg;
    m.other = hold;
    return fgChild_Message(*self, msg);
  }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgWindow";
  case FG_SETNAME:
    if(self->name) free(self->name);
    self->name = fgCopyText((const char*)msg->other);
    fgSendMsg<FG_SETSKIN, void*>(*self, 0); // force the skin to be recalculated
    return 0;
  case FG_GETNAME:
    return (size_t)self->name;
  }
  return fgChild_Message(*self, msg);
}

size_t FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg)
{
  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_MOUSEON:
    fgSendMsg<FG_HOVER>(*self);
    break;
  case FG_MOUSEUP:
    if(MsgHitCRect(msg, &self->element)) // We can get a MOUSEUP when the mouse is outside of the control but we DO NOT fire it unless it's actually in the control.
      fgSendMsg<FG_ACTION>(*self);
    fgSendMsg<FG_HOVER>(*self); // Revert to hover no matter what. The other handler will fire off a mousemove for us that will handle the hover change event.
    if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgCaptureWindow = 0;
    break;
  case FG_MOUSEOFF:
    fgSendMsg<FG_NUETRAL>(*self);
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      fgCaptureWindow = *self;
      fgSendMsg<FG_ACTIVE>(*self);
    }
    break;
  }
  return fgWindow_Message(self,msg);
}

FG_EXTERN void FG_FASTCALL fgWindow_TabAfter(fgWindow* self, fgWindow* prev)
{
  self->tabnext = prev->tabnext;
  self->tabprev = prev;
  prev->tabnext = self;
}

FG_EXTERN void FG_FASTCALL fgWindow_TabBefore(fgWindow* self, fgWindow* next)
{
  self->tabprev = next->tabprev;
  self->tabnext = next;
  next->tabprev = self;
}