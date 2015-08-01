// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgRoot.h"
#include <stdlib.h>

void FG_FASTCALL fgRoot_Init(fgRoot* self)
{
  fgWindow_Init((fgWindow*)self,0,0,0,0);
  self->gui.element.destroy=&fgRoot_Destroy;
  self->gui.message=&fgRoot_Message;
  self->behaviorhook=&fgRoot_BehaviorDefault;
  self->update=&fgRoot_Update;
  self->updateroot=0;
  self->keymsghook=&fgRoot_KeyMsgHook;
  self->time=0;
  self->winrender=&fgRoot_WinRender;
  fgWindow_Init(&self->mouse,0,0,0,0);
}

void FG_FASTCALL fgRoot_Destroy(fgRoot* self)
{
  (*self->mouse.element.destroy)(&self->mouse);
  fgWindow_Destroy((fgWindow*)self);
}

char FG_FASTCALL fgRoot_Message(fgWindow* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_GOTFOCUS:
    return 1; //Root cannot have focus
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgRoot";
    return 0;
  }
  return fgWindow_Message(self,msg);
}

// Recursive event injection function
char FG_FASTCALL fgRoot_RInject(fgRoot* root, fgWindow* self, const FG_Msg* msg, AbsRect* area)
{
  AbsRect curarea;
  fgChild* cur = self->element.root;
  while(cur && (cur->flags&FGWIN_NOCLIP)!=0) // Go through all our children that aren't being clipped
  {
    if(!fgRoot_RInject(root,(fgWindow*)cur,msg,area)) //pass through the parent area because these aren't clipped
      return 0; // If the message is NOT rejected, return 0 immediately to indicate we ate the message.
    cur=cur->next; // Otherwise the child rejected the message.
  }
  assert(msg!=0 && area!=0);
  ResolveRectCache(&curarea,&self->element,area);
  if((self->element.flags&FGWIN_HIDDEN)!=0 || !MsgHitAbsRect(msg,&curarea)) //If the event completely misses us, or we're hidden, we must reject it
    return 1;

  while(cur) // Try to inject to any children we have
  {
    if(!fgRoot_RInject(root,(fgWindow*)cur,msg,&curarea)) // If the message is NOT rejected, return 0 immediately.
      return 0;
    cur=cur->next; // Otherwise the child rejected the message.
  } // If we get this far either we have no children, the event missed them all, or they all rejected the event.

  return (*root->behaviorhook)((fgWindow*)self,msg);
}

char FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg)
{
  static AbsRect absarea = { 0,0,0,0 };
  CRect* mousearea;
  assert(self!=0);

  switch(msg->type)
  {
  case FG_KEYCHAR:
  case FG_KEYUP:
  case FG_KEYDOWN:
  case FG_MOUSESCROLL:
    if(!(*self->keymsghook)(msg))
      return 0;
    if(fgFocusedWindow)
      return (*self->behaviorhook)(fgFocusedWindow,msg);
    break;
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
    mousearea=&self->mouse.element.element.area; // Update our mouse-tracking static
    mousearea->left.abs=mousearea->right.abs=(FABS)msg->x;
    mousearea->top.abs=mousearea->bottom.abs=(FABS)msg->y;

    if(fgFocusedWindow)
      if(!(*self->behaviorhook)(fgFocusedWindow,msg))
        return 1;

    if(!fgRoot_RInject(self,(fgWindow*)self,msg,&absarea))
      break; // it's important to break, not return, because we need to reset nonclip
    
    if(msg->type==FG_MOUSEMOVE && fgLastHover!=0) // If we STILL haven't accepted a mousemove event, send a MOUSEOFF message if lasthover exists
    {
      fgWindow_BasicMessage(fgLastHover,FG_MOUSEOFF);
      fgLastHover=0;
    }
    return 1;
  }
  return 0;
}

char FG_FASTCALL fgRoot_BehaviorDefault(fgWindow* self, const FG_Msg* msg)
{
  assert(self!=0);
  return (*self->message)(self,msg);
}
char FG_FASTCALL fgRoot_CallBehavior(fgWindow* self, const FG_Msg* msg)
{
  return (*fgSingleton()->behaviorhook)(self,msg);
}

void FG_FASTCALL fgTerminate(fgRoot* root)
{
  VirtualFreeChild((fgChild*)root);
}

void FG_FASTCALL fgRoot_RListRender(fgStatic* self, AbsRect* area)
{
  AbsRect curarea;
  assert(area!=0);
  while(self)
  {
    if(!(self->element.flags&FGSTATIC_HIDDEN))
    {
      ResolveRectCache(&curarea,&self->element,area);
      // TODO: allow children to be drawn before parent if the order is negative
      (*self->message)(self,FG_RDRAW,&curarea,0);
      if(self->element.last) // Render everything backwards
        fgRoot_RListRender((fgStatic*)self->element.last,&curarea);
    }
    self=(fgStatic*)self->element.prev;
  }
}

void FG_FASTCALL fgRoot_WinRender(fgWindow* self, AbsRect* area)
{
  AbsRect curarea;
  fgChild* cur = self->element.last;
  assert(area!=0);
  ResolveRectCache(&curarea,&self->element,area);

  if(self->element.flags&FGWIN_HIDDEN) // If we aren't visible, none of our children are either, so bail out.
    return;
  if(self->rlast)
    fgRoot_RListRender(self->rlast,&curarea);
  
  while(cur) // Render all our children (backwards)
  {
    fgRoot_WinRender((fgWindow*)cur,&curarea);
    cur=cur->prev; 
  } 
}

void FG_FASTCALL fgRoot_Render(fgRoot* self)
{
  static AbsRect absarea = { 0,0,0,0 };

  assert(self!=0);
  (*self->winrender)((fgWindow*)self, &absarea);
}

void FG_FASTCALL fgRoot_Update(fgRoot* self, double delta)
{
  fgDeferAction* cur;
  self->time+=delta;

  while((cur=self->updateroot) && (cur->time<=self->time))
  {
    self->updateroot=cur->next;
    if((*cur->action)(cur->arg)) // If this returns true, we deallocate the node
      free(cur);
  }
}
fgDeferAction* FG_FASTCALL fgRoot_AllocAction(char (FG_FASTCALL *action)(void*), void* arg, double time)
{
  fgDeferAction* r = (fgDeferAction*)malloc(sizeof(fgDeferAction));
  r->action=action;
  r->arg=arg;
  r->time=time;
  r->next=0; // We do this so its never ambigious if an action is in the list already or not
  r->prev=0;
  return r;
}
void FG_FASTCALL fgRoot_DeallocAction(fgRoot* self, fgDeferAction* action)
{
  if(action->prev!=0 || action==self->updateroot) // If true you are in the list and must be removed
    fgRoot_RemoveAction(self,action);
  free(action);
}

void FG_FASTCALL fgRoot_AddAction(fgRoot* self, fgDeferAction* action)
{
  fgDeferAction* cur = self->updateroot;
  fgDeferAction* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(action!=0 && !action->prev && action!=self->updateroot);
  while(cur != 0 && cur->time < action->time)
  {
     prev = cur;
     cur = cur->next;
  }
  action->next = cur;
  action->prev = prev;
  if(prev) prev->next=action;
  else self->updateroot=action; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev=action; // Cur is null if we are at the end of the list.
}
void FG_FASTCALL fgRoot_RemoveAction(fgRoot* self, fgDeferAction* action)
{
  assert(action!=0 && (action->prev!=0 || action==self->updateroot));
	if(action->prev != 0) action->prev->next = action->next;
  else self->updateroot = action->next;
	if(action->next != 0) action->next->prev = action->prev;
  action->next=0; // We do this so its never ambigious if an action is in the list already or not
  action->prev=0;
}
void FG_FASTCALL fgRoot_ModifyAction(fgRoot* self, fgDeferAction* action)
{
  if((action->next!=0 && action->next->time<action->time) || (action->prev!=0 && action->prev->time>action->time))
  {
    fgRoot_RemoveAction(self,action);
    fgRoot_AddAction(self,action);
  }
  else if(!action->prev && action!=self->updateroot) // If true you aren't in the list so we need to add you
    fgRoot_AddAction(self,action);
}
char FG_FASTCALL fgRoot_KeyMsgHook(const FG_Msg* msg)
{
  return 1;
}
