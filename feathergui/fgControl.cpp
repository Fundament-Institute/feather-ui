// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgControl.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "fgMenu.h"

fgElement* fgFocusedWindow = 0;
fgElement* fgLastHover = 0;
fgElement* fgCaptureWindow = 0;

void fgControl_Init(fgControl* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, units, (fgDestroy)&fgControl_Destroy, (fgMessage)&fgControl_Message);
}

void fgControl_Destroy(fgControl* self)
{
  assert(self != 0);
  if(self->tabprev) self->tabprev->tabnext = self->tabnext;
  if(self->tabnext) self->tabnext->tabprev = self->tabprev;
  if(self->sideprev) self->sideprev->sidenext = self->sidenext;
  if(self->sidenext) self->sidenext->sideprev = self->sideprev;
  self->tabnext = self->tabprev = 0;
  self->sidenext = self->sideprev = 0;

  self->element.message = (fgMessage)fgElement_Message; // Setting the message function to the element message function after the destructor mimics destroying this class's virtual functions, as C++ would have done.
  fgElement_Destroy(&self->element);
}

void fgElement_DoHoverCalc(fgElement* self) // This is a seperate function so some controls can force a focus event to instead activate hover instead.
{
  if(fgLastHover != self)
  {
    if(fgLastHover != 0)
      _sendmsg<FG_MOUSEOFF>(fgLastHover);
    fgLastHover = self;
    _sendmsg<FG_MOUSEON>(self);
  }
}

size_t fgControl_Message(fgControl* self, const FG_Msg* msg)
{
  assert(self != 0);
  assert(msg != 0);
  fgFlag otherint = (fgFlag)msg->u;

  if(self->element.flags&FGCONTROL_DISABLE)
  { // If this control is disabled, it absorbs all input messages that reach it without responding to them.
    switch(msg->type)
    {
    case FG_MOUSEDOWN:
    case FG_MOUSEMOVE:
      fgElement_DoHoverCalc(*self);
      return FGCURSOR_ARROW;
    case FG_MOUSEDBLCLICK:
    case FG_MOUSEUP:
    case FG_MOUSEON:
    case FG_MOUSEOFF:
    case FG_MOUSESCROLL:
    case FG_TOUCHBEGIN:
    case FG_TOUCHEND:
    case FG_TOUCHMOVE:
    case FG_KEYUP:
    case FG_KEYDOWN:
    case FG_KEYCHAR:
    case FG_JOYBUTTONDOWN:
    case FG_JOYBUTTONUP:
    case FG_JOYAXIS:
    case FG_GOTFOCUS:
    case FG_LOSTFOCUS:
      return FG_ACCEPT;
    }
  }

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(*self, msg);
    self->contextmenu = 0;
    self->tabnext = self->tabprev = self; // This creates an infinite loop of tabbing
    self->sidenext = self->sideprev = self;
    fgStandardNeutralSetStyle(*self, "neutral");
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgControl* hold = reinterpret_cast<fgControl*>(msg->e);
      memsubset<fgControl, fgElement>(hold, 0);
      hold->contextmenu = self->contextmenu;
      fgElement_Message(&self->element, msg);
    }
    return sizeof(fgControl);
  case FG_KEYDOWN:
  {
    fgControl* target = 0;
    switch(msg->keycode)
    {
    case FG_KEY_LEFT:
    case FG_KEY_RIGHT:
      target = (msg->keycode == FG_KEY_LEFT) ? self->sideprev : self->sidenext;
      break;
    case FG_KEY_UP:
    case FG_KEY_DOWN:
      target = (msg->keycode == FG_KEY_UP) ? self->tabprev : self->tabnext;
      break;
    case FG_KEY_TAB:
      if(msg->sigkeys&(2 | 4))
        return 0;
      target = (msg->sigkeys & 1) ? self->tabprev : self->tabnext;
      break;
    default:
      return 0;
    }
    if(target != 0 && target != self)
      _sendmsg<FG_GOTFOCUS>(*target);
  }
    return FG_ACCEPT;
  case FG_MOUSEDOWN:
    fgElement_DoHoverCalc(*self);
    if(fgFocusedWindow != *self)
      _sendmsg<FG_GOTFOCUS>(*self);
    return FG_ACCEPT;
  case FG_MOUSEMOVE:
    fgElement_DoHoverCalc(*self);
    return FGCURSOR_ARROW;
  case FG_MOUSEUP:
    if(msg->button == FG_MOUSERBUTTON && self->contextmenu != 0)
    {
      AbsRect area = { 0 };
      if(self->contextmenu->parent)
        ResolveRect(self->contextmenu->parent, &area);
      MoveCRect(msg->x - area.left, msg->y - area.top, &self->contextmenu->transform.area);
      fgMenu_Show((fgMenu*)self->contextmenu, true);
      fgCaptureWindow = self->contextmenu;
    }
  { // Any control that gets a MOUSEUP event immediately fires a MOUSEMOVE event at that location, which will force the focus to shift to a different control if the mouseup occured elsewhere.
    FG_Msg m = *msg;
    m.type = FG_MOUSEMOVE;
    fgroot_instance->inject(fgroot_instance, &m);
  }
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->element.flags&FGCONTROL_DISABLE)
      return 0;
    if(fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      _sendmsg<FG_LOSTFOCUS, void*>(fgFocusedWindow, self);
    fgFocusedWindow = *self;
    fgSetFlagStyle(*self, "focused", true);
    return FG_ACCEPT;
  case FG_LOSTFOCUS:
    assert(fgFocusedWindow == *self);
    if(fgFocusedWindow == *self)
    {
      fgFocusedWindow = 0;
      fgSetFlagStyle(*self, "focused", false);
      if(self->element.parent)
      {
        if(self->element.parent->lastfocus == *self) // if the lastfocus was already us set it to 0.
          self->element.parent->lastfocus = 0;
        if(!msg->p)
          _sendmsg<FG_GOTFOCUS>(self->element.parent);
        else
          self->element.parent->lastfocus = *self;
      }
    }
    return FG_ACCEPT;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->element.flags, otherint, msg->u2);
  case FG_SETFLAGS:
  {
    bool disabled = !(self->element.flags & FGCONTROL_DISABLE) && ((otherint & FGCONTROL_DISABLE) != 0);
    fgElement_Message(*self, msg); // apply the flag change first or we'll get weird bugs.
    if(disabled)
    {
      if(fgFocusedWindow == *self)
        _sendmsg<FG_LOSTFOCUS, void*>(*self, 0);
      if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
        fgCaptureWindow = 0;
      if(fgLastHover == *self)
      {
        _sendmsg<FG_MOUSEOFF>(*self);
        fgLastHover = 0;
      }
    }
  }
    return FG_ACCEPT;
  case FG_SETCONTEXTMENU:
    self->contextmenu = msg->e;
    break;
  case FG_GETCONTEXTMENU:
    return (size_t)self->contextmenu;
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Control";
  }
  return fgElement_Message(*self, msg);
}

size_t fgControl_HoverMessage(fgControl* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  if(self->element.flags&FGCONTROL_DISABLE)
  { // If this control is disabled, it absorbs all input messages that reach it without responding to them.
    switch(msg->type)
    {
    case FG_MOUSEDOWN:
    case FG_MOUSEMOVE:
      fgElement_DoHoverCalc(*self);
    case FG_MOUSEDBLCLICK:
    case FG_MOUSEUP:
    case FG_MOUSEON:
    case FG_MOUSEOFF:
    case FG_MOUSESCROLL:
    case FG_TOUCHBEGIN:
    case FG_TOUCHEND:
    case FG_TOUCHMOVE:
    case FG_KEYUP:
    case FG_KEYDOWN:
    case FG_KEYCHAR:
    case FG_JOYBUTTONDOWN:
    case FG_JOYBUTTONUP:
    case FG_JOYAXIS:
    case FG_GOTFOCUS:
    case FG_LOSTFOCUS:
      return FG_ACCEPT;
    }
  }

  switch(msg->type)
  {
  case FG_KEYDOWN:
    if(msg->keycode == FG_KEY_RETURN)
    {
      _sendmsg<FG_ACTION>(*self);
      return FG_ACCEPT;
    }
    break;
  case FG_MOUSEON:
    _sendmsg<FG_HOVER>(*self);
    break;
  case FG_MOUSEUP:
    _sendmsg<FG_HOVER>(*self); // Revert to hover no matter what. The other handler will fire off a mousemove for us that will handle the hover change event.
    if(fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgCaptureWindow = 0;
    break;
  case FG_MOUSEOFF:
    _sendmsg<FG_NEUTRAL>(*self);
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

size_t fgControl_ActionMessage(fgControl* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  if(!(self->element.flags&FGCONTROL_DISABLE))
  {
    switch(msg->type)
    {
    case FG_MOUSEUP:
      if(MsgHitElement(msg, &self->element) && fgCaptureWindow == &self->element) // We can get a MOUSEUP when the mouse is outside of the control but we DO NOT fire it unless it's actually in the control.
      {
        size_t r = fgControl_HoverMessage(self, msg);
        _sendmsg<FG_ACTION>(*self);
        return r;
      }
      break;
    }
  }

  return fgControl_HoverMessage(self, msg);
}

void fgControl_TabAfter(fgControl* self, fgControl* prev)
{
  self->tabnext = prev->tabnext;
  self->tabprev = prev;
  prev->tabnext = self;
}

void fgControl_TabBefore(fgControl* self, fgControl* next)
{
  self->tabprev = next->tabprev;
  self->tabnext = next;
  next->tabprev = self;
}