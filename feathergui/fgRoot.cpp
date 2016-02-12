// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgRoot.h"
#include <stdlib.h>
#include "feathercpp.h"

fgRoot* fgroot_instance = 0;

void FG_FASTCALL fgRoot_Init(fgRoot* self)
{
  self->behaviorhook = &fgRoot_BehaviorDefault;
  self->drag = 0;
  self->time = 0.0;
  self->updateroot = 0;
  self->radiohash = fgRadioGroup_init();
  fgroot_instance = self;
  fgChild_InternalSetup((fgChild*)self, 0, 0, 0, (FN_DESTROY)&fgRoot_Destroy, (FN_MESSAGE)&fgRoot_Message);
}

void FG_FASTCALL fgRoot_Destroy(fgRoot* self)
{
  fgRadioGroup_destroy(self->radiohash);
  fgWindow_Destroy((fgWindow*)self);
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
    return 1; 
  case FG_GETCLASSNAME:
    return (size_t)"fgRoot";
  case FG_DRAW:
  {
    CRect* rootarea = &self->gui.element.element.area;
    AbsRect area = { rootarea->left.abs, rootarea->top.abs, rootarea->right.abs, rootarea->bottom.abs };
    FG_Msg m = *msg;
    m.other = &area;
    return fgWindow_Message((fgWindow*)self, &m);
  }
  }
  return fgWindow_Message((fgWindow*)self,msg);
}

void FG_FASTCALL fgStandardDraw(fgChild* self, AbsRect* area, int max)
{
  fgChild* hold = self->last; // we draw backwards through our list.
  AbsRect curarea;
  char clipping = 0;

  while(hold && hold->order <= max)
  {
    if(!(hold->flags&FGCHILD_HIDDEN))
    {
      if(!clipping && !(hold->flags&FGCHILD_NOCLIP))
      {
        clipping = 1;
        fgPushClipRect(area);
      }
      else if(clipping && (hold->flags&FGCHILD_NOCLIP))
      {
        clipping = 0;
        fgPopClipRect();
      }

      ResolveRectCache(&curarea, hold, area);
      fgChild_VoidMessage(hold, FG_DRAW, &curarea);
    }
    hold = hold->prev;
  }

  if(clipping)
    fgPopClipRect();
}

// Recursive event injection function
size_t FG_FASTCALL fgRoot_RInject(fgRoot* root, fgChild* self, const FG_Msg* msg, AbsRect* area)
{
  AbsRect curarea;
  fgChild* cur = self->rootnoclip;
  while(cur) // Go through all our children that aren't being clipped
  {
    if(!fgRoot_RInject(root,cur,msg,area)) //pass through the parent area because these aren't clipped
      return 0; // If the message is NOT rejected, return 0 immediately to indicate we accepted the message.
    cur=cur->next; // Otherwise the child rejected the message.
  }
  assert(msg!=0 && area!=0);
  ResolveRectCache(&curarea,self,area);
  if((self->flags&FGCHILD_HIDDEN)!=0 || !MsgHitAbsRect(msg,&curarea)) //If the event completely misses us, or we're hidden, we must reject it
    return 1;

  cur = self->rootclip;
  while(cur) // Try to inject to any children we have
  {
    if(!fgRoot_RInject(root,cur,msg,&curarea)) // If the message is NOT rejected, return 0 immediately.
      return 0;
    cur=cur->next; // Otherwise the child rejected the message.
  }

  // If we get this far either we have no children, the event missed them all, or they all rejected the event...
  return (*root->behaviorhook)(self,msg); // So we give the event to ourselves
}

size_t FG_FASTCALL fgRoot_Inject(fgRoot* self, const FG_Msg* msg)
{
  assert(self != 0);
  CRect* rootarea = &self->gui.element.element.area;
  AbsRect absarea = { rootarea->left.abs, rootarea->top.abs, rootarea->right.abs, rootarea->bottom.abs };

  switch(msg->type)
  {
  case FG_JOYBUTTONDOWN:
  case FG_JOYBUTTONUP:
  case FG_JOYAXIS:
  case FG_KEYCHAR:
  case FG_KEYUP:
  case FG_KEYDOWN:
  {
    fgChild* cur = !fgFocusedWindow ? (fgChild*)self : fgFocusedWindow;
    do
    {
      if(!(*self->behaviorhook)(cur, msg))
        return 0;
      cur = cur->parent;
    } while(cur);
    return 1;
  }
  case FG_MOUSESCROLL:
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEMOVE:
    if(self->drag && self->drag->parent == (fgChild*)self)
    {
      AbsVec pos = { (FABS)msg->x, (FABS)msg->y };
      MoveCRect(pos, &self->drag->element.area);
    }
    if(fgCaptureWindow)
      if(!(*self->behaviorhook)(fgCaptureWindow,msg))
        return 0;

    if(!fgRoot_RInject(self, (fgChild*)self, msg, &absarea))
      return 0;
    if(msg->type != FG_MOUSEMOVE)
      break;
  case FG_MOUSELEAVE:
    if(fgLastHover!=0) // If we STILL haven't accepted a mousemove event, send a MOUSEOFF message if lasthover exists
    {
      fgChild_VoidMessage(fgLastHover, FG_MOUSEOFF, 0);
      fgLastHover=0;
    }
    break;
  }
  return 1;
}

size_t FG_FASTCALL fgRoot_BehaviorDefault(fgChild* self, const FG_Msg* msg)
{
  assert(self!=0);
  return (*self->message)(self,msg);
}

void FG_FASTCALL fgTerminate(fgRoot* root)
{
  VirtualFreeChild((fgChild*)root);
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
fgRoot* FG_FASTCALL fgSingleton()
{
  return fgroot_instance;
}