// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgControl.h"
#include "fgRoot.h"
#include "feathercpp.h"
#include "fgMenu.h"

void fgControl_Init(fgControl* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
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

  if(self->tooltip)
    fgFreeText(self->tooltip, __FILE__, __LINE__);
  if(self->tooltipaction)
    fgDeallocAction(self->tooltipaction);

  self->element.message = (fgMessage)fgElement_Message; // Setting the message function to the element message function after the destructor mimics destroying this class's virtual functions, as C++ would have done.
  fgElement_Destroy(&self->element);
}

void fgElement_DoHoverCalc(fgElement* self) // This is a seperate function so some controls can force a focus event to instead activate hover instead.
{
  if(fgroot_instance->fgLastHover != self)
  {
    if(fgroot_instance->fgLastHover != 0)
      _sendmsg<FG_MOUSEOFF>(fgroot_instance->fgLastHover);
    fgroot_instance->fgLastHover = self;
    _sendmsg<FG_MOUSEON>(self);
  }
}

void fgElement_TriggerContextMenu(fgElement* contextmenu, const FG_Msg* msg)
{
  AbsRect area = { 0 };
  if(contextmenu->parent)
    ResolveRect(contextmenu->parent, &area);
  MoveCRect(msg->x - area.left, msg->y - area.top, &contextmenu->transform.area);
  fgMenu_Show((fgMenu*)contextmenu, true, false);
  fgroot_instance->fgCaptureWindow = contextmenu;
}

char fgControl_TooltipAction(void* p)
{
  fgRoot_ShowTooltip(fgroot_instance, reinterpret_cast<fgControl*>(p)->tooltip);
  return 0; // We don't want to deallocate the node
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
    self->tooltip = 0;
    self->tooltipaction = fgAllocAction(&fgControl_TooltipAction, self, fgroot_instance->time);
    self->tabnext = self->tabprev = self; // This creates an infinite loop of tabbing
    self->sidenext = self->sideprev = self;
    (*self)->SetStyle((self->element.flags & FGCONTROL_DISABLE) ? "disabled" : "neutral");
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgControl* hold = reinterpret_cast<fgControl*>(msg->e);
      bss::memsubset<fgControl, fgElement>(hold, 0);
      hold->contextmenu = self->contextmenu;
      hold->tabnext = hold->tabprev = hold;
      hold->sidenext = hold->sideprev = hold;
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
    if(self->tooltip)
    {
      fgRemoveAction(self->tooltipaction);
      fgRoot_ShowTooltip(fgroot_instance, 0);
    }
    fgElement_DoHoverCalc(*self);
    if(fgroot_instance->fgFocusedWindow != *self)
      _sendmsg<FG_GOTFOCUS>(*self);
    return FG_ACCEPT;
  case FG_MOUSEMOVE:
    fgElement_DoHoverCalc(*self);
    return FGCURSOR_ARROW;
  case FG_MOUSEUP:
    if(msg->button == FG_MOUSERBUTTON && self->contextmenu != 0)
      fgElement_TriggerContextMenu(self->contextmenu, msg);
  { // Any control that gets a MOUSEUP event immediately fires a MOUSEMOVE event at that location, which will force the focus to shift to a different control if the mouseup occured elsewhere.
    FG_Msg m = *msg;
    m.type = FG_MOUSEMOVE;
    fgroot_instance->inject(fgroot_instance, &m);
  }
    return FG_ACCEPT;
  case FG_GOTFOCUS:
    if(self->element.flags&FGCONTROL_DISABLE)
      return 0;
    fgroot_instance->fgPendingFocusedWindow = &self->element; // We have to track our pending focused window so we can detect if the LOSTFOCUS event destroys this control.
    if(fgroot_instance->fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      _sendmsg<FG_LOSTFOCUS, void*>(fgroot_instance->fgFocusedWindow, self);
    if(fgroot_instance->fgPendingFocusedWindow == &self->element)
    {
      fgroot_instance->fgFocusedWindow = *self;
      fgSetFlagStyle(&self->element, "focused", true);
      fgroot_instance->fgPendingFocusedWindow = 0;
    }
    return FG_ACCEPT;
  case FG_LOSTFOCUS:
    assert(fgroot_instance->fgFocusedWindow == *self);
    if(fgroot_instance->fgFocusedWindow == *self)
    {
      fgroot_instance->fgFocusedWindow = 0;
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
    otherint = bss::bssSetBit<fgFlag>(self->element.flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((self->element.flags ^ otherint) & FGCONTROL_DISABLE)
    {
      if((otherint&FGCONTROL_DISABLE) && fgroot_instance->fgLastHover == *self) // if we were hovered, remove hover before adding disable flag (or we'll never get the message)
      {
        _sendmsg<FG_MOUSEOFF>(*self);
        fgroot_instance->fgLastHover = 0;
      }
      fgElement_Message(*self, msg); // apply the flag change first or we'll get weird bugs.
      if(otherint & FGCONTROL_DISABLE) // Check to see if the disabled flag was added
      {
        if(fgroot_instance->fgFocusedWindow == *self)
          _sendmsg<FG_LOSTFOCUS, void*>(*self, 0);
        if(fgroot_instance->fgCaptureWindow == *self) // Remove our control hold on mouse messages.
          fgroot_instance->fgCaptureWindow = 0;
      }
      (*self)->SetStyle((otherint & FGCONTROL_DISABLE) ? "disabled" : "neutral");
      fgroot_instance->mouse.state |= FGMOUSE_SEND_MOUSEMOVE;
      return FG_ACCEPT;
    }
    break;
  case FG_SETCONTEXTMENU:
    self->contextmenu = msg->e;
    break;
  case FG_GETCONTEXTMENU:
    return (size_t)self->contextmenu;
    break;
  case FG_SETTOOLTIP:
    if(self->tooltip)
      fgFreeText(self->tooltip, __FILE__, __LINE__);
    self->tooltip = fgCopyTextToUTF8(msg->p, msg->subtype, __FILE__, __LINE__);
    return self->tooltip != 0;
  case FG_MOUSEON:
    if(self->tooltip)
    {
      self->tooltipaction->time = fgroot_instance->time + fgroot_instance->tooltipdelay;
      fgAddAction(self->tooltipaction);
    }
    break;
  case FG_MOUSEOFF:
    if(self->tooltip)
    {
      fgRemoveAction(self->tooltipaction);
      fgRoot_ShowTooltip(fgroot_instance, 0);
    }
    break;
  case FG_GETTOOLTIP:
    return (size_t)self->tooltip;
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
    if(fgroot_instance->fgCaptureWindow == *self) // Remove our control hold on mouse messages.
      fgroot_instance->fgCaptureWindow = 0;
    break;
  case FG_MOUSEOFF:
    _sendmsg<FG_NEUTRAL>(*self);
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON)
    {
      fgroot_instance->fgCaptureWindow = *self;
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
      if(MsgHitElement(msg, &self->element) && fgroot_instance->fgCaptureWindow == &self->element) // We can get a MOUSEUP when the mouse is outside of the control but we DO NOT fire it unless it's actually in the control.
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