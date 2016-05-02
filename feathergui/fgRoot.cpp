// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgRoot.h"
#include "fgMonitor.h"
#include "feathercpp.h"
#include <stdlib.h>

fgRoot* fgroot_instance = 0;

void FG_FASTCALL fgRoot_Init(fgRoot* self, const AbsRect* area, size_t dpi)
{
  self->behaviorhook = &fgRoot_BehaviorDefault;
  self->dpi = dpi;
  self->drag = 0;
  self->monitors = 0;
  self->time = 0.0;
  self->updateroot = 0;
  self->radiohash = fgRadioGroup_init();
  fgroot_instance = self;
  fgTransform transform = { area->left, 0, area->top, 0, area->right, 0, area->bottom, 0, 0, 0, 0 };
  fgElement_InternalSetup(*self, 0, 0, 0, 0, &transform, (FN_DESTROY)&fgRoot_Destroy, (FN_MESSAGE)&fgRoot_Message);
}

void FG_FASTCALL fgRoot_Destroy(fgRoot* self)
{
  fgRadioGroup_destroy(self->radiohash);
  fgControl_Destroy((fgControl*)self);
}

void fgRoot_BuildMouseMove(fgRoot* self, FG_Msg* msg)
{
}

void FG_FASTCALL fgRoot_CheckMouseMove(fgRoot* self)
{
  if(self->mouse.state&FGMOUSE_SEND_MOUSEMOVE)
  {
    self->mouse.state &= ~FGMOUSE_SEND_MOUSEMOVE;
    FG_Msg m = { 0 };
    m.type = FG_MOUSEMOVE;
    m.x = self->mouse.x;
    m.y = self->mouse.y;
    m.allbtn = self->mouse.buttons;
    fgRoot_Inject(self, &m);
  }
}

size_t FG_FASTCALL fgRoot_Message(fgRoot* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_KEYCHAR: // If these messages get sent to the root, they have been rejected from everything else.
  case FG_KEYUP:
  case FG_KEYDOWN:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
  case FG_MOUSESCROLL:
  case FG_GOTFOCUS: //Root cannot have focus
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgRoot";
  case FG_DRAW:
  {
	fgRoot_CheckMouseMove(self);
	CRect* rootarea = &self->gui.element.transform.area;
    AbsRect area = { rootarea->left.abs, rootarea->top.abs, rootarea->right.abs, rootarea->bottom.abs };
    FG_Msg m = *msg;
    m.other = &area;
    return fgControl_Message((fgControl*)self, &m);
  }
  case FG_GETDPI:
    return self->dpi;
  case FG_SETDPI:
  {
    float scale = !self->dpi ? 1.0f : (msg->otherint / (float)self->dpi);
    CRect* rootarea = &self->gui.element.transform.area;
    CRect area = { rootarea->left.abs*scale, 0, rootarea->top.abs*scale, 0, rootarea->right.abs*scale, 0, rootarea->bottom.abs*scale, 0 };
    self->dpi = (size_t)msg->otherint;
    return self->gui.element.SetArea(area);
  }
    return 1;
  }
  return fgControl_Message((fgControl*)self,msg);
}

void FG_FASTCALL fgStandardDraw(fgElement* self, AbsRect* area, size_t dpi, char culled)
{
  fgElement* hold = culled ? self->rootnoclip : self->root;
  AbsRect curarea;
  bool clipping = false;

  while(hold)
  {
    if(!(hold->flags&FGELEMENT_HIDDEN))
    {
      ResolveRectCache(hold, &curarea, area, (hold->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);

      if(!clipping && !(hold->flags&FGELEMENT_NOCLIP))
      {
        clipping = true;
        fgPushClipRect(area);
      }
      else if(clipping && (hold->flags&FGELEMENT_NOCLIP))
      {
        clipping = false;
        fgPopClipRect();
      }

      char culled = !fgRectIntersect(&curarea, &fgPeekClipRect());
      _sendsubmsg<FG_DRAW, void*, size_t>(hold, culled, &curarea, dpi);
    }
    hold = hold->next;
  }

  if(clipping)
    fgPopClipRect();
}

// Recursive event injection function
size_t FG_FASTCALL fgRoot_RInject(fgRoot* root, fgElement* self, const FG_Msg* msg, AbsRect* area, AbsRect* padding)
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // IF this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND) ? 0 : padding);

  fgElement* cur = self->lastnoclip;
  while(cur) // Go through all our children that aren't being clipped
  {
    if(fgRoot_RInject(root, cur, msg, &curarea, &self->padding)) // We still need to properly evaluate hitboxes even for nonclipping elements.
      return 1; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
    cur=cur->prev; // Otherwise the child rejected the message.
  }

  if(area != 0 && !MsgHitAbsRect(msg, &curarea)) // If the event completely misses us, we must reject it. If the area is null, we always accept the message.
    return 0;

  cur = self->lastclip;
  while(cur) // Try to inject to any children we have
  {
    if(fgRoot_RInject(root, cur, msg, &curarea, &self->padding)) // If the message is NOT rejected, return 0 immediately.
      return 1;
    cur = cur->prev; // Otherwise the child rejected the message.
  }

  // If we get this far either we have no children, the event missed them all, or they all rejected the event...
  return (*root->behaviorhook)(self,msg); // So we give the event to ourselves
}

size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg)
{
  assert(self != 0);

  CRect* rootarea = &self->gui.element.transform.area;
  fgUpdateMouseState(&self->mouse, msg);

  switch(msg->type)
  {
  case FG_JOYBUTTONDOWN:
  case FG_JOYBUTTONUP:
  case FG_JOYAXIS:
  case FG_KEYCHAR:
  case FG_KEYUP:
  case FG_KEYDOWN:
  {
    fgElement* cur = !fgFocusedWindow ? *self : fgFocusedWindow;
    do
    {
      if((*self->behaviorhook)(cur, msg))
        return 1;
      cur = cur->parent;
    } while(cur);
    return 0;
  }
  case FG_MOUSESCROLL:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
    if(self->drag != 0 && self->drag->parent == *self)
      MoveCRect((FABS)msg->x, (FABS)msg->y, &self->drag->transform.area);

    if(fgCaptureWindow)
      if(fgRoot_RInject(self, fgCaptureWindow, msg, 0, 0)) // If it's captured, send the message to the captured window with NULL area.
        return 1;

    if(fgRoot_RInject(self, *self, msg, 0, 0))
      return 1;
    if(msg->type != FG_MOUSEMOVE)
      break;
  case FG_MOUSELEAVE:
    if(fgLastHover != 0) // If we STILL haven't accepted a mousemove event, send a MOUSEOFF message if lasthover exists
    {
      _sendmsg<FG_MOUSEOFF>(fgLastHover);
      fgLastHover = 0;
    }
    break;
  }
  return 0;
}

size_t FG_FASTCALL fgRoot_BehaviorDefault(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  return (*self->message)(self, msg);
}

FG_EXTERN size_t FG_FASTCALL fgRoot_BehaviorListener(fgElement* self, const FG_Msg* msg)
{
  assert(self != 0);
  size_t ret = (*self->message)(self, msg);
  khiter_t iter = fgListenerHash.Iterator(std::pair<fgElement*, unsigned short>(self, msg->type));
  if(fgListenerHash.ExistsIter(iter))
    fgListenerHash.GetValue(iter)(self, msg);
  return ret;
}

void FG_FASTCALL fgTerminate(fgRoot* root)
{
  VirtualFreeChild((fgElement*)root);
}

void FG_FASTCALL fgRoot_Update(fgRoot* self, double delta)
{
  fgDeferAction* cur;
  self->time += delta;

  while((cur = self->updateroot) && (cur->time <= self->time))
  {
    self->updateroot = cur->next;
    if((*cur->action)(cur->arg)) // If this returns true, we deallocate the node
      free(cur);
  }
}

FG_EXTERN fgMonitor* FG_FASTCALL fgRoot_GetMonitor(const fgRoot* self, const AbsRect* rect)
{
  fgMonitor* cur = self->monitors;
  float largest = 0;
  fgMonitor* last = 0; // it is possible for all intersections ot have an area of zero, meaning the element is not on ANY monitors and is thus not visible.

  while(cur)
  {
    CRect& area = cur->element.transform.area;
    AbsRect monitor = { area.left.abs > rect->left ? area.left.abs : rect->left,
      area.top.abs > rect->top ? area.top.abs : rect->top,
      area.right.abs < rect->right ? area.right.abs : rect->right,
      area.bottom.abs < rect->bottom ? area.bottom.abs : rect->bottom };

    float total = (monitor.right - monitor.left)*(monitor.bottom - monitor.top);
    if(total > largest) // This must be GREATER THAN to ensure that a value of "0" is not ever assigned a monitor.
    {
      largest = total;
      last = cur;
    }

    cur = cur->mnext;
  }

  return last;
}

fgDeferAction* FG_FASTCALL fgRoot_AllocAction(char (FG_FASTCALL *action)(void*), void* arg, double time)
{
  fgDeferAction* r = (fgDeferAction*)malloc(sizeof(fgDeferAction));
  r->action = action;
  r->arg = arg;
  r->time = time;
  r->next = 0; // We do this so its never ambigious if an action is in the list already or not
  r->prev = 0;
  return r;
}

void FG_FASTCALL fgRoot_DeallocAction(fgRoot* self, fgDeferAction* action)
{
  if(action->prev != 0 || action == self->updateroot) // If true you are in the list and must be removed
    fgRoot_RemoveAction(self, action);
  free(action);
}

void FG_FASTCALL fgRoot_AddAction(fgRoot* self, fgDeferAction* action)
{
  fgDeferAction* cur = self->updateroot;
  fgDeferAction* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(action != 0 && !action->prev && action != self->updateroot);
  while(cur != 0 && cur->time < action->time)
  {
    prev = cur;
    cur = cur->next;
  }
  action->next = cur;
  action->prev = prev;
  if(prev) prev->next = action;
  else self->updateroot = action; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev = action; // Cur is null if we are at the end of the list.
}

void FG_FASTCALL fgRoot_RemoveAction(fgRoot* self, fgDeferAction* action)
{
  assert(action != 0 && (action->prev != 0 || action == self->updateroot));
  if(action->prev != 0) action->prev->next = action->next;
  else self->updateroot = action->next;
  if(action->next != 0) action->next->prev = action->prev;
  action->next = 0; // We do this so its never ambigious if an action is in the list already or not
  action->prev = 0;
}

void FG_FASTCALL fgRoot_ModifyAction(fgRoot* self, fgDeferAction* action)
{
  if((action->next != 0 && action->next->time < action->time) || (action->prev != 0 && action->prev->time > action->time))
  {
    fgRoot_RemoveAction(self, action);
    fgRoot_AddAction(self, action);
  }
  else if(!action->prev && action != self->updateroot) // If true you aren't in the list so we need to add you
    fgRoot_AddAction(self, action);
}

fgRoot* FG_FASTCALL fgSingleton()
{
  return fgroot_instance;
}
