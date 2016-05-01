// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgControl.h"
#include "fgRoot.h"
#include "feathercpp.h"

fgElement* fgFocusedWindow = 0;
fgElement* fgLastHover = 0;
fgElement* fgCaptureWindow = 0;

void FG_FASTCALL fgControl_Init(fgControl* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, (FN_DESTROY)&fgControl_Destroy, (FN_MESSAGE)&fgControl_Message);
}

void FG_FASTCALL fgControl_Destroy(fgControl* self)
{
  assert(self != 0);
  if(self->tabprev) self->tabprev->tabnext = self->tabnext;
  if(self->tabnext) self->tabnext->tabprev = self->tabprev;
  if(self->sideprev) self->sideprev->sidenext = self->sidenext;
  if(self->sidenext) self->sidenext->sideprev = self->sideprev;
  self->tabnext = self->tabprev = 0;
  self->sidenext = self->sideprev = 0;

  fgElement_Destroy(&self->element);
}

void FG_FASTCALL fgControl_DoHoverCalc(fgControl* self) // This is a seperate function so some controls can force a focus event to instead activate hover instead.
{
  if(fgLastHover != *self)
  {
    if(fgLastHover != 0)
      _sendmsg<FG_MOUSEOFF>(fgLastHover);
    fgLastHover = *self;
    _sendmsg<FG_MOUSEON>(*self);
  }
}

size_t FG_FASTCALL fgControl_Message(fgControl* self, const FG_Msg* msg)
{
  assert(self != 0);
  assert(msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(*self, msg);
    self->contextmenu = 0;
    self->tabnext = self->tabprev = self; // This creates an infinite loop of tabbing
    self->sidenext = self->sideprev = self;
    return 1;
  case FG_KEYDOWN:
    if(msg->keycode == FG_KEY_TAB && !(msg->sigkeys&(2 | 4)))
    {
      fgControl* target = (msg->sigkeys & 2) ? self->tabprev : self->tabnext;
      if(target != 0 && target != self)
        _sendmsg<FG_GOTFOCUS>(*target);
      return 1;
    }
    break;
  case FG_MOUSEDOWN:
    fgControl_DoHoverCalc(self);
    if(fgFocusedWindow != *self)
      _sendmsg<FG_GOTFOCUS>(*self);
    //if(msg->button == FG_MOUSERBUTTON && self->contextmenu != 0)
    //  _sendmsg<FG_GOTFOCUS>(*self->contextmenu);
    return 1;
  case FG_MOUSESCROLL:
    fgControl_DoHoverCalc(self);
    return 1;
  case FG_MOUSEMOVE:
    fgControl_DoHoverCalc(self);
    if(fgroot_instance->drag) // Send a dragging message if necessary. Does not initiate a drag for you (this is because some drags are initiated via click and drag, and some are just clicking).
    {
      FG_Msg m = *msg;
      m.type = FG_DRAGGING;
      fgPassMessage(*self, &m);
    }
    return 1;
  case FG_MOUSEUP:
  { // Any control that gets a MOUSEUP event immediately fires a MOUSEMOVE event at that location, which will force the focus to shift to a different control if the mouseup occured elsewhere.
    FG_Msg m = *msg;
    m.type = FG_MOUSEMOVE;
    fgRoot_Inject(fgroot_instance, &m);
  }
  if(fgroot_instance->drag != 0) // If necessary, send a drop message to the current control we're hovering over.
  {
    _sendmsg<FG_DROP, void*>(fgLastHover, fgroot_instance->drag);
    fgroot_instance->drag = 0; // Ensure the drag pointer is set to NULL even if the target window doesn't understand the FG_DROP message.
  }
  return 1;
  case FG_GOTFOCUS:
    if(fgElement_Message(*self, msg)) // checks if we resolved via lastfocus.
      return 1;
    if(fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      _sendmsg<FG_LOSTFOCUS, void*>(fgFocusedWindow, self);
    fgFocusedWindow = *self;
    return 1;
  case FG_LOSTFOCUS:
    assert(fgFocusedWindow == *self);
    if(fgFocusedWindow == *self)
    {
      fgFocusedWindow = 0;
      if(self->element.parent)
      {
        if(!msg->other)
          _sendmsg<FG_GOTFOCUS>(self->element.parent);
        else
          self->element.parent->lastfocus = *self;
      }
    }
    return 1;
  case FG_CLONE:
  {
    fgControl* hold = (fgControl*)msg->other;
    if(!hold)
      hold = (fgControl*)malloc(sizeof(fgControl));
    hold->contextmenu = hold->contextmenu;

    FG_Msg m = *msg;
    m.other = hold;
    return fgElement_Message(*self, msg);
  }
  return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgControl";
  }
  return fgElement_Message(*self, msg);
}

size_t FG_FASTCALL fgControl_HoverMessage(fgControl* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_MOUSEON:
    _sendmsg<FG_HOVER>(*self);
    break;
  case FG_MOUSEUP:
    if(MsgHitCRect(msg, &self->element)) // We can get a MOUSEUP when the mouse is outside of the control but we DO NOT fire it unless it's actually in the control.
      _sendmsg<FG_ACTION>(*self);
    _sendmsg<FG_HOVER>(*self); // Revert to hover no matter what. The other handler will fire off a mousemove for us that will handle the hover change event.
    if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgCaptureWindow = 0;
    break;
  case FG_MOUSEOFF:
    _sendmsg<FG_NUETRAL>(*self);
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      fgCaptureWindow = *self;
      _sendmsg<FG_ACTIVE>(*self);
    }
    break;
  }
  return fgControl_Message(self, msg);
}

FG_EXTERN void FG_FASTCALL fgControl_TabAfter(fgControl* self, fgControl* prev)
{
  self->tabnext = prev->tabnext;
  self->tabprev = prev;
  prev->tabnext = self;
}

FG_EXTERN void FG_FASTCALL fgControl_TabBefore(fgControl* self, fgControl* next)
{
  self->tabprev = next->tabprev;
  self->tabnext = next;
  next->tabprev = self;
}