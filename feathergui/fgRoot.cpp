// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgRoot.h"
#include "fgMonitor.h"
#include "fgBox.h"
#include "fgWindow.h"
#include "fgRadiobutton.h"
#include "fgProgressbar.h"
#include "fgSlider.h"
#include "fgTextbox.h"
#include "fgTreeview.h"
#include "fgDebug.h"
#include "fgList.h"
#include "fgCurve.h"
#include "fgDropdown.h"
#include "fgTabcontrol.h"
#include "fgMenu.h"
#include "feathercpp.h"
#include "bss-util/cTrie.h"
#include <stdlib.h>
#include <sstream>

KHASH_INIT(fgIDMap, char*, fgElement*, 1, kh_str_hash_func, kh_str_hash_equal);
KHASH_INIT(fgIDHash, fgElement*, char*, 1, kh_ptr_hash_func, kh_int_hash_equal);

fgRoot* fgroot_instance = 0;

void FG_FASTCALL fgRoot_Init(fgRoot* self, const AbsRect* area, size_t dpi, fgBackend* backend)
{
  static fgBackend DEFAULT_BACKEND = {
    &fgBehaviorHookDefault,
    &fgCreateFontDefault,
    &fgCopyFontDefault,
    &fgCloneFontDefault,
    &fgDestroyFontDefault,
    &fgDrawFontDefault,
    &fgFontSizeDefault,
    &fgFontGetDefault,
    &fgCreateResourceDefault,
    &fgCloneResourceDefault,
    &fgDestroyResourceDefault,
    &fgDrawResourceDefault,
    &fgResourceSizeDefault,
    &fgFontIndexDefault,
    &fgFontPosDefault,
    &fgDrawLinesDefault,
    &fgCreateDefault,
    &fgMessageMapDefault,
    &fgUserDataMapDefault,
    &fgPushClipRectDefault,
    &fgPeekClipRectDefault,
    &fgPopClipRectDefault,
    &fgDragStartDefault,
    &fgSetCursorDefault,
    &fgClipboardCopyDefault,
    &fgClipboardExistsDefault,
    &fgClipboardPasteDefault,
    &fgClipboardFreeDefault,
    &fgDirtyElementDefault,
    0,
    0,
  };

  memset(self, 0, sizeof(fgRoot));
  self->backend = !backend ? DEFAULT_BACKEND : *backend;
  self->dpi = dpi;
  self->cursorblink = 0.53; // 530 ms is the windows default.
  self->lineheight = 30;
  self->fontscale = 1.0f;
  self->radiohash = fgRadioGroup_init();
  self->functionhash = fgFunctionMap_init();
  self->idhash = kh_init_fgIDHash();
  self->idmap = kh_init_fgIDMap();
  fgroot_instance = self;
  fgTransform transform = { area->left, 0, area->top, 0, area->right, 0, area->bottom, 0, 0, 0, 0 };
  fgElement_InternalSetup(*self, 0, 0, 0, 0, &transform, 0, (fgDestroy)&fgRoot_Destroy, (fgMessage)&fgRoot_Message);
  self->gui.element.style = 0;
}

void FG_FASTCALL fgRoot_Destroy(fgRoot* self)
{
  if(fgdebug_instance != 0)
    VirtualFreeChild(&fgdebug_instance->element);
  fgRadioGroup_destroy(self->radiohash);
  fgFunctionMap_destroy(self->functionhash);
  fgControl_Destroy((fgControl*)self);
  kh_destroy_fgIDHash(self->idhash);
  kh_destroy_fgIDMap(self->idmap);
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
  case FG_MOUSEMOVE:
    fgRoot_SetCursor(FGCURSOR_ARROW, 0);
  case FG_KEYCHAR: // If these messages get sent to the root, they have been rejected from everything else.
  case FG_KEYUP:
  case FG_KEYDOWN:
  case FG_MOUSEOFF:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSESCROLL:
  case FG_GOTFOCUS: //Root cannot have focus
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"Root";
  case FG_DRAW:
  {
	fgRoot_CheckMouseMove(self);
	CRect* rootarea = &self->gui.element.transform.area;
    AbsRect area = { rootarea->left.abs, rootarea->top.abs, rootarea->right.abs, rootarea->bottom.abs };
    FG_Msg m = *msg;
    m.other = &area;
    fgControl_Message((fgControl*)self, &m);
    if(self->topmost) // Draw topmost before the drag object
    {
      AbsRect out;
      ResolveRect(self->topmost, &out);
      self->topmost->Draw(&out, (int)self->topmost->GetDPI());
    }
    if(self->dragdraw != 0 && self->dragdraw->parent != *self)
    {
      AbsRect out;
      ResolveRect(self->dragdraw, &out);
      FABS dx = out.right - out.left;
      FABS dy = out.bottom - out.top;
      out.left = (FABS)self->mouse.x;
      out.top = (FABS)self->mouse.y;
      out.right = out.left + dx;
      out.bottom = out.top + dy;
      self->dragdraw->Draw(&out, (int)self->dragdraw->GetDPI());
    }
    return FG_ACCEPT;
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
  case FG_GETSTYLE:
    return 0;
  }
  return fgControl_Message((fgControl*)self,msg);
}

void BSS_FORCEINLINE fgStandardApplyClipping(fgFlag flags, const AbsRect* area, bool& clipping)
{
  if(!clipping && !(flags&FGELEMENT_NOCLIP))
  {
    clipping = true;
    fgroot_instance->backend.fgPushClipRect(area);
  }
  else if(clipping && (flags&FGELEMENT_NOCLIP))
  {
    clipping = false;
    fgroot_instance->backend.fgPopClipRect();
  }
}

void BSS_FORCEINLINE fgStandardDrawElement(fgElement* self, fgElement* hold, const AbsRect* area, size_t dpi, AbsRect& curarea, bool& clipping)
{
  if(!(hold->flags&FGELEMENT_HIDDEN) && hold != fgroot_instance->topmost)
  {
    ResolveRectCache(hold, &curarea, area, (hold->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
    fgStandardApplyClipping(hold->flags, area, clipping);

    char culled = !fgRectIntersect(&curarea, &fgroot_instance->backend.fgPeekClipRect());
    _sendsubmsg<FG_DRAW, void*, size_t>(hold, culled, &curarea, dpi);
  }
}

void FG_FASTCALL fgStandardDraw(fgElement* self, const AbsRect* area, size_t dpi, char culled)
{
  fgElement* hold = culled ? self->rootnoclip : self->root;
  AbsRect curarea;
  bool clipping = false;

  while(hold)
  {
    fgStandardDrawElement(self, hold, area, dpi, curarea, clipping);
    hold = culled ? hold->nextnoclip : hold->next;
  }

  if(clipping)
    fgroot_instance->backend.fgPopClipRect();
}

void FG_FASTCALL fgOrderedDraw(fgElement* self, const AbsRect* area, size_t dpi, char culled, fgElement* skip, fgElement* (*fn)(fgElement*, const AbsRect*), void(*draw)(fgElement*, const AbsRect*, size_t))
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

  if(draw)
    draw(self, area, dpi);

  AbsRect out;
  fgRectIntersection(area, &fgroot_instance->backend.fgPeekClipRect(), &out);
  // do binary search on the absolute resolved bottomright coordinates compared to the topleft corner of the render area
  cur = fn(self, &out);

  if(!clipping)
  {
    clipping = true; // always clipping at this stage because ordered elements can't be nonclipping
    fgroot_instance->backend.fgPushClipRect(area);
  }
  char cull = 0;

  while(!cull && cur != 0 && !(cur->flags & FGELEMENT_BACKGROUND)) // Render all ordered elements until they become culled
  {
    ResolveRectCache(cur, &curarea, area, &self->padding); // always apply padding because these are always foreground elements
    cull = !fgRectIntersect(&curarea, &fgroot_instance->backend.fgPeekClipRect());
    if(cur != fgroot_instance->topmost)
      _sendsubmsg<FG_DRAW, void*, size_t>(cur, cull, &curarea, dpi);
    cur = cur->next;
  }

  cur = skip;
  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND)) // Render all background elements after the ordered elements
  {
    fgStandardDrawElement(self, cur, area, dpi, curarea, clipping);
    cur = cur->next;
  }

  if(clipping)
    fgroot_instance->backend.fgPopClipRect();
}

void FG_FASTCALL fgFixedDraw(fgElement* self, AbsRect* area, size_t dpi, char culled, fgElement** ordered, size_t numordered, AbsVec dim)
{

}


// Recursive event injection function
size_t FG_FASTCALL fgStandardInject(fgElement* self, const FG_Msg* msg, const AbsRect* area)
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // If this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND || !self->parent) ? 0 : &self->parent->padding);

  bool miss = (area != 0 && !MsgHitAbsRect(msg, &curarea)); // If the area is null, the message always hits.
  fgElement* cur = miss ? self->lastnoclip : self->lastinject; // If the event completely misses us, evaluate only nonclipping elements.
  while(cur) // Try to inject to any children we have
  {
    if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)) // We have to check FGELEMENT_IGNORE because the noclip list may have render-only elements in it.
      return FG_ACCEPT; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
    cur = miss ? cur->prevnoclip : cur->previnject; // Otherwise the child rejected the message.
  }

  // If we get this far either we have no children, the event missed them all, or they all rejected the event...
  return miss ? 0 : (*fgroot_instance->backend.behaviorhook)(self,msg); // So we give the event to ourselves, but only if it didn't miss us (which can happen if we were evaluating nonclipping elements)
}

size_t FG_FASTCALL fgOrderedInject(fgElement* self, const FG_Msg* msg, const AbsRect* area, fgElement* skip, fgElement* (*fn)(fgElement*, const FG_Msg*))
{
  assert(msg != 0);

  if((self->flags&FGELEMENT_HIDDEN) != 0) // If we're hidden we always reject messages no matter what.
    return 0;

  AbsRect curarea;
  if(!area) // If this is null either we are the root or this is a captured message, in which case we would have to resolve the entire relative coordinate chain anyway
    ResolveRect(self, &curarea);
  else
    ResolveRectCache(self, &curarea, area, (self->flags & FGELEMENT_BACKGROUND || !self->parent) ? 0 : &self->parent->padding);

  if(area != 0 && !MsgHitAbsRect(msg, &curarea)) // if this misses us, only evaluate nonclipping elements. Don't bother with the ordered array.
  {
    fgElement* cur = self->lastnoclip;
    while(cur) // Try to inject to any children we have
    {
      if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea)) // We have to check FGELEMENT_IGNORE because the noclip list may have render-only elements in it.
        return FG_ACCEPT; // If the message is NOT rejected, return 1 immediately to indicate we accepted the message.
      cur = cur->prevnoclip; // Otherwise the child rejected the message.
    }

    return 0; // We either had no nonclipping children or it missed them all, so reject this.
  }

  fgElement* cur = self->lastinject;

  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND))
  {
    if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea))
      return FG_ACCEPT; 
    cur = cur->previnject; 
  }

  cur = fn(self, msg);
  if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea))
    return FG_ACCEPT;

  cur = skip;
  while(cur != 0 && (cur->flags & FGELEMENT_BACKGROUND))
  {
    if(!(cur->flags&FGELEMENT_IGNORE) && _sendmsg<FG_INJECT, const void*, const void*>(cur, msg, &curarea))
      return FG_ACCEPT;
    cur = cur->previnject;
  }

  return (*fgroot_instance->backend.behaviorhook)(self, msg); // So we give the event to ourselves because it couldn't have missed us if we got to this point
}

void fgProcessNextCursor(fgRoot* self)
{
  if(self->overridecursor != FGCURSOR_NONE)
  {
    self->nextcursor = self->overridecursor;
    self->nextcursordata = self->overridecursordata;
  }

  if(self->nextcursor != self->lastcursor || self->nextcursordata != self->lastcursordata)
  {
    self->backend.fgSetCursor(self->nextcursor, self->nextcursordata);
    self->lastcursor = self->nextcursor;
    self->lastcursordata = self->lastcursordata;
  }
}

size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg)
{
  assert(self != 0);

  CRect* rootarea = &self->gui.element.transform.area;
  fgUpdateMouseState(&self->mouse, msg);
  FG_Msg m;

  if(self->dragtype != FGCLIPBOARD_NONE && (msg->type == FG_MOUSEMOVE || msg->type == FG_MOUSEUP))
  {
    m = *msg;
    m.type = (msg->type == FG_MOUSEMOVE) ? FG_DRAGOVER : FG_DROP;
    msg = &m;
  }

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
      if((*self->backend.behaviorhook)(cur, msg))
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
    if(self->dragdraw != 0 && self->dragdraw->parent == *self)
      MoveCRect((FABS)msg->x, (FABS)msg->y, &self->dragdraw->transform.area);

    if(fgCaptureWindow)
      if(_sendmsg<FG_INJECT, const void*, const void*>(fgCaptureWindow, msg, 0)) // If it's captured, send the message to the captured window with NULL area.
        return fgProcessNextCursor(self), FG_ACCEPT;

    if(self->topmost) // After we attempt sending the message to the captured window, try sending it to the topmost
      if(_sendmsg<FG_INJECT, const void*, const void*>(self->topmost, msg, 0))
        return fgProcessNextCursor(self), FG_ACCEPT;

    if(_sendmsg<FG_INJECT, const void*, const void*>(*self, msg, 0))
      return fgProcessNextCursor(self), FG_ACCEPT;
    if(msg->type != FG_MOUSEMOVE)
      break;
    fgProcessNextCursor(self);
  case FG_MOUSEOFF:
    if(fgLastHover != 0) // If we STILL haven't accepted a mousemove event, send a MOUSEOFF message if lasthover exists
    {
      _sendmsg<FG_MOUSEOFF>(fgLastHover);
      fgLastHover = 0;
    }
    if(msg->type != FG_MOUSEOFF)
      break;
  case FG_MOUSEON:
    self->lastcursor = FGCURSOR_NONE;
    break;
  case FG_DRAGOVER:
  case FG_DROP:
    _sendmsg<FG_INJECT, const void*, const void*>(*self, msg, 0);
    fgProcessNextCursor(self);
    if(msg->type == FG_DROP)
    {
      self->dragtype = FGCLIPBOARD_NONE;
      if(self->dragdraw != 0)
      {
        if(self->dragdraw->parent == *self)
          VirtualFreeChild(self->dragdraw);
        self->dragdraw = 0;
      }
      self->dragdata = 0;
    }
    break;
  }
  return 0;
}

void FG_FASTCALL fgRoot_SetCursor(char cursor, void* data)
{
  if(cursor & FGCURSOR_OVERRIDE)
  {
    fgroot_instance->overridecursor = cursor;
    fgroot_instance->overridecursordata = data;
  }
  else
  {
    fgroot_instance->nextcursor = cursor;
    fgroot_instance->nextcursordata = data;
  }
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
      fgfree(cur, __FILE__, __LINE__);
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
  fgDeferAction* r = fgmalloc<fgDeferAction>(1, __FILE__, __LINE__);
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
  fgfree(action, __FILE__, __LINE__);
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
fgElement* FG_FASTCALL fgRoot_GetID(fgRoot* self, const char* id)
{
  khiter_t i = kh_get_fgIDMap(self->idmap, const_cast<char*>(id));
  if(i != kh_end(self->idmap) && kh_exist(self->idmap, i))
    return kh_val(self->idmap, i);
  return 0;
}

#ifdef BSS_DEBUG
void VERIFY_IDHASH()
{
  for(khiter_t i = 0; i < fgroot_instance->idhash->n_buckets; ++i)
    if(i != kh_end(fgroot_instance->idhash) && kh_exist(fgroot_instance->idhash, i))
      assert(kh_val(fgroot_instance->idhash, i) != (void*)0xcdcdcdcdcdcdcdcd);
}
#endif

void FG_FASTCALL fgRoot_AddID(fgRoot* self, const char* id, fgElement* element)
{
  int r;
  khiter_t i = kh_get_fgIDMap(self->idmap, const_cast<char*>(id));
  if(i != kh_end(self->idmap) && kh_exist(self->idmap, i))  // the key already exists so we need to remove and replace the previous element
  {
    assert(i != kh_end(self->idmap));
    kh_del_fgIDHash(self->idhash, kh_get_fgIDHash(self->idhash, kh_val(self->idmap, i)));
  }
  else
    i = kh_put_fgIDMap(self->idmap, fgCopyText(id, __FILE__, __LINE__), &r);

  kh_val(self->idmap, i) = element;
  khiter_t j = kh_put_fgIDHash(self->idhash, element, &r);
  const char* test = kh_key(self->idmap, i);
  kh_val(self->idhash, j) = kh_key(self->idmap, i);
}
bool FG_FASTCALL fgRoot_RemoveID(fgRoot* self, fgElement* element)
{
  khiter_t i = kh_get_fgIDHash(self->idhash, element);
  if(i == kh_end(self->idhash) || !kh_exist(self->idhash, i))
    return false;
  khiter_t j = kh_get_fgIDMap(self->idmap, kh_val(self->idhash, i));
  if(j != kh_end(self->idmap) && kh_exist(self->idmap, j))
    kh_del_fgIDMap(self->idmap, j);
  else
    assert(false);

  fgfree(kh_val(self->idhash, i), __FILE__, __LINE__);
  kh_del_fgIDHash(self->idhash, i);
  return true;
}

fgRoot* FG_FASTCALL fgSingleton()
{
  return fgroot_instance;
}

fgElement* FG_FASTCALL fgCreate(const char* type, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  return fgroot_instance->backend.fgCreate(type, parent, next, name, flags, transform, units);
}
