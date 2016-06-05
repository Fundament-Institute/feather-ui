// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgRoot.h"
#include "fgMonitor.h"
#include "fgBox.h"
#include "fgWindow.h"
#include "fgRadioButton.h"
#include "fgProgressbar.h"
#include "fgSlider.h"
#include "fgTextbox.h"
#include "fgTreeView.h"
#include "fgDebug.h"
#include "feathercpp.h"
#include "bss-util/cTrie.h"
#include <stdlib.h>
#include <sstream>

fgRoot* fgroot_instance = 0;

void FG_FASTCALL fgRoot_Init(fgRoot* self, const AbsRect* area, size_t dpi)
{
  self->behaviorhook = &fgRoot_BehaviorDefault;
  self->dpi = dpi;
  self->drag = 0;
  self->monitors = 0;
  self->time = 0.0;
  self->updateroot = 0;
  self->lineheight = 0;
  self->radiohash = fgRadioGroup_init();
  fgroot_instance = self;
  fgTransform transform = { area->left, 0, area->top, 0, area->right, 0, area->bottom, 0, 0, 0, 0 };
  fgElement_InternalSetup(*self, 0, 0, 0, 0, &transform, (fgDestroy)&fgRoot_Destroy, (fgMessage)&fgRoot_Message);
}

void FG_FASTCALL fgRoot_Destroy(fgRoot* self)
{
  fgRadioGroup_destroy(self->radiohash);
  fgControl_Destroy((fgControl*)self);
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
  case FG_MOUSEDBLCLICK:
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
    return FG_ACCEPT;
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->otherf;
    return FG_ACCEPT;
  }
  return fgControl_Message((fgControl*)self,msg);
}

void BSS_FORCEINLINE fgStandardDrawElement(fgElement* self, fgElement* hold, AbsRect* area, size_t dpi, AbsRect& curarea, bool& clipping)
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
}

void FG_FASTCALL fgStandardDraw(fgElement* self, AbsRect* area, size_t dpi, char culled)
{
  fgElement* hold = culled ? self->rootnoclip : self->root;
  AbsRect curarea;
  bool clipping = false;

  while(hold)
  {
    fgStandardDrawElement(self, hold, area, dpi, curarea, clipping);
    hold = culled ? hold->nextclip : hold->next;
  }

  if(clipping)
    fgPopClipRect();
}

void FG_FASTCALL fgOrderedDraw(fgElement* self, AbsRect* area, size_t dpi, char culled, fgElement** ordered, size_t numordered)
{
  if(culled) // If we are culled, thee's no point drawing ordered elements, because ordered elements aren't non-clipping, so we let the standard draw take care of it.
    return fgStandardDraw(self, area, dpi, culled);

  fgElement* cur = self->root;
  AbsRect curarea;
  bool clipping = false;

  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND)) // Render all background elements before the ordered elements
  {
    fgStandardDrawElement(self, cur, area, dpi, curarea, clipping);
    cur = cur->next;
  }

  // do binary search on the absolute resolved bottomright coordinates compared to the topleft corner of the render area
  cur = 0;

  clipping = true; // always clipping at this stage
  fgPushClipRect(area);
  char cull = 0;

  while(!cull && cur != 0 && !(cur->flags & FGELEMENT_BACKGROUND)) // Render all ordered elements until they become culled
  {
    ResolveRectCache(cur, &curarea, area, &self->padding); // always apply padding because these are always foreground elements
    cull = !fgRectIntersect(&curarea, &fgPeekClipRect());
    _sendsubmsg<FG_DRAW, void*, size_t>(cur, cull, &curarea, dpi);
  }

  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND)) // Render all background elements after the ordered elements
  {
    fgStandardDrawElement(self, cur, area, dpi, curarea, clipping);
    cur = cur->next;
  }

  if(clipping)
    fgPopClipRect();
}


// Recursive event injection function
size_t FG_FASTCALL fgStandardInject(fgElement* self, const FG_Msg* msg, const AbsRect* area)
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // IF this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND || !self->parent) ? 0 : &self->parent->padding);

  fgElement* cur = self->lastnoclip;
  while(cur) // Go through all our children that aren't being clipped
  {
    if(_sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)) // We still need to properly evaluate hitboxes even for nonclipping elements.
      return FG_ACCEPT; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
    cur=cur->prev; // Otherwise the child rejected the message.
  }

  if(area != 0 && !MsgHitAbsRect(msg, &curarea)) // If the event completely misses us, we must reject it. If the area is null, we always accept the message.
    return 0;

  cur = self->lastclip;
  while(cur) // Try to inject to any children we have
  {
    if(_sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)) // If the message is NOT rejected, return 0 immediately.
      return FG_ACCEPT;
    cur = cur->prev; // Otherwise the child rejected the message.
  }

  // If we get this far either we have no children, the event missed them all, or they all rejected the event...
  return (*fgroot_instance->behaviorhook)(self,msg); // So we give the event to ourselves
}

size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg)
{
  assert(self != 0);

  CRect* rootarea = &self->gui.element.transform.area;
  fgUpdateMouseState(&self->mouse, msg);

  switch(msg->type)
  {
  case FG_KEYUP:
  case FG_KEYDOWN:
    self->keys[msg->keycode / 32] = (msg->type == FG_KEYDOWN) ? (self->keys[msg->keycode / 32] | (1 << (msg->keycode % 32))) : (self->keys[msg->keycode / 32] & (~(1 << (msg->keycode % 32))));
  case FG_JOYBUTTONDOWN:
  case FG_JOYBUTTONUP:
  case FG_JOYAXIS:
  case FG_KEYCHAR:
  {
    fgElement* cur = !fgFocusedWindow ? *self : fgFocusedWindow;
    do
    {
      if((*self->behaviorhook)(cur, msg))
        return FG_ACCEPT;
      cur = cur->parent;
    } while(cur);
    return 0;
  }
  case FG_MOUSESCROLL:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEDOWN:
    rootarea = rootarea;
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
    if(self->drag != 0 && self->drag->parent == *self)
      MoveCRect((FABS)msg->x, (FABS)msg->y, &self->drag->transform.area);

    if(fgCaptureWindow)
      if(_sendmsg<FG_INJECT, const void*, const void*>(fgCaptureWindow, msg, 0)) // If it's captured, send the message to the captured window with NULL area.
        return FG_ACCEPT;

    if(_sendmsg<FG_INJECT, const void*, const void*>(*self, msg, 0))
      return FG_ACCEPT;
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

size_t FG_FASTCALL fgRoot_BehaviorListener(fgElement* self, const FG_Msg* msg)
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

fgMonitor* FG_FASTCALL fgRoot_GetMonitor(const fgRoot* self, const AbsRect* rect)
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
  fgDeferAction* r = bss_util::bssmalloc<fgDeferAction>(1);
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

template<class T, void (FG_FASTCALL *INIT)(T* BSS_RESTRICT, fgElement* BSS_RESTRICT, fgElement* BSS_RESTRICT, const char*, fgFlag, const fgTransform*)>
fgElement* _create_default(fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  T* r = bss_util::bssmalloc<T>(1);
  INIT(r, parent, next, name, flags, transform);
  return (fgElement*)r;
}

fgElement* FG_FASTCALL fgCreateDefault(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  static bss_util::cTrie<uint16_t> t(15, "fgelement", "fgcontrol", "fgresource", "fgtext", "fgbox", "fgscrollbar", "fgbutton", "fgwindow", "fgcheckbox",
    "fgradiobutton", "fgprogressbar", "fgslider", "fgtextbox", "fgtreeview", "fgtreeitem", "fgdebug");
  
  size_t len = strlen(type) + 1; // include null terminator
  DYNARRAY(char, lower, len);
  STRNCPY(lower, len, type, len);
  STRLWR(lower);
  
  switch(t[lower])
  {
  case 0:
    return _create_default<fgElement, fgElement_Init>(parent, next, name, flags, transform);
  case 1:
    return _create_default<fgControl, fgControl_Init>(parent, next, name, flags, transform);
  case 2:
    return _create_default<fgResource, fgResource_Init>(parent, next, name, flags, transform);
  case 3:
    return _create_default<fgText, fgText_Init>(parent, next, name, flags, transform);
  case 4:
    return _create_default<fgBox, fgBox_Init>(parent, next, name, flags, transform);
  case 5:
    return _create_default<fgScrollbar, fgScrollbar_Init>(parent, next, name, flags, transform);
  case 6:
    return _create_default<fgButton, fgButton_Init>(parent, next, name, flags, transform);
  case 7:
    return _create_default<fgWindow, fgWindow_Init>(parent, next, name, flags, transform);
  case 8:
    return _create_default<fgCheckbox, fgCheckbox_Init>(parent, next, name, flags, transform);
  case 9:
    return _create_default<fgRadiobutton, fgRadiobutton_Init>(parent, next, name, flags, transform);
  case 10:
    return _create_default<fgProgressbar, fgProgressbar_Init>(parent, next, name, flags, transform);
  case 11:
    return _create_default<fgSlider, fgSlider_Init>(parent, next, name, flags, transform);
  case 12:
    return _create_default<fgTextbox, fgTextbox_Init>(parent, next, name, flags, transform);
  case 13:
    return _create_default<fgTreeView, fgTreeView_Init>(parent, next, name, flags, transform);
  case 14:
    return _create_default<fgTreeItem, fgTreeItem_Init>(parent, next, name, flags, transform);
  case 15:
    return _create_default<fgDebug, fgDebug_Init>(parent, next, name, flags, transform);
  }

  return 0;
}