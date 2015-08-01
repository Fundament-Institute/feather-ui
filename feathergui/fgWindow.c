// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWindow.h"
#include "fgRoot.h"
#include "fgSkin.h"

fgWindow* fgFocusedWindow = 0;
fgWindow* fgLastHover = 0;

void FG_FASTCALL fgWindow_Init(fgWindow* BSS_RESTRICT self, fgWindow* BSS_RESTRICT parent, const fgElement* element, FG_UINT id, fgFlag flags)
{ 
  assert(self!=0);
  memset(self,0,sizeof(fgWindow)); 
  self->element.destroy=&fgWindow_Destroy; 
  self->element.free=&free;
  self->message=&fgWindow_Message;
  self->id=id;
  self->element.flags=flags;
  if(element) self->element.element=*element;
  if(parent) fgWindow_VoidMessage(parent,FG_ADDCHILD,self);
}
void FG_FASTCALL fgWindow_Destroy(fgWindow* self)
{
  fgStatic* cur;
  assert(self!=0);
  fgWindow_VoidMessage(self,FG_SETPARENT,0);
  while((cur=self->rlist)!=0)
    VirtualFreeChild((fgChild*)cur); // Removes static from rlist

  fgVector_Destroy(&self->skinstatics);
  if(self->name) free(self->name);
  fgChild_Destroy(&self->element);
}

char FG_FASTCALL fgWindow_Message(fgWindow* self, const FG_Msg* msg)
{
  int otherint = msg->otherint;
  fgChild* hold;
  assert(self!=0);
  assert(msg!=0);
  
  switch(msg->type)
  {
  default:
    return 1;
  case FG_MOUSEUP: // This is done to take care of the "click button, hold mouse, move over another button, release" problem.
  case FG_MOUSEMOVE:
    if(MsgHitCRect(msg,&self->element))
    {
      if(fgLastHover!=self)
      {
        if(fgLastHover!=0)
          fgWindow_BasicMessage(fgLastHover,FG_MOUSEOFF);
        fgLastHover=self;
        fgWindow_BasicMessage(self,FG_MOUSEON);
      }
      return 0;
    }
    return 1;
  case FG_MOUSEDOWN:
    if(fgFocusedWindow != self)
      fgWindow_BasicMessage(self,FG_GOTFOCUS);
    if(msg->button == 2 && self->contextmenu!=0)
      fgWindow_BasicMessage((fgWindow*)self->contextmenu,FG_GOTFOCUS);
    return !MsgHitCRect(msg,&self->element);
  case FG_GOTFOCUS:
    if(fgFocusedWindow) // We do this here so you can disable getting focus by blocking this message without messing things up
      fgWindow_BasicMessage(fgFocusedWindow,FG_LOSTFOCUS);
    fgFocusedWindow=self;
    break;
  case FG_LOSTFOCUS:
    assert(fgFocusedWindow==self);
    fgFocusedWindow=0;
    break;
  case FG_ADDCHILD:
    hold=(fgChild*)msg->other;
    if(!hold) return 1;
    if(hold->flags&FGSTATIC_MARKER)
    {
      if(hold->parent!=0)
        fgStatic_NotifyParent((fgStatic*)hold);
      fgStatic_SetParent((fgStatic*)msg->other,&self->element);
    }
    else 
    {
      assert(hold!=0 && hold->destroy!=fgChild_Destroy);
      assert(hold->parent==0 || hold->parent->destroy!=fgChild_Destroy);
      if(hold->parent!=0)
        fgWindow_VoidMessage((fgWindow*)hold->parent,FG_REMOVECHILD,hold);
      fgWindow_SetParent((fgWindow*)msg->other,&self->element);
    }
    break;
  case FG_REMOVECHILD:
    hold=(fgChild*)msg->other;
    if(!hold) return 1;
    assert(hold->parent==(fgChild*)self);
    if(hold->flags&FGSTATIC_MARKER)
      fgStatic_SetParent((fgStatic*)hold,0);
    else
      fgWindow_SetParent((fgWindow*)hold,0);
    break;
  case FG_SETPARENT:
    hold=(fgChild*)msg->other;
    if(hold==self->element.parent) break;
    assert(hold==0 || hold->destroy!=fgChild_Destroy);
    if(hold!=0) {
      if(hold->flags&FGSTATIC_MARKER) return 1;
      fgWindow_VoidMessage((fgWindow*)hold,FG_ADDCHILD,self); // This will remove us from our parent if we have one.
    } else // If hold is zero, then our parent MUST be nonzero, or they'd both be 0 and we wouldn't have reached this code.
      fgWindow_VoidMessage((fgWindow*)self->element.parent,FG_REMOVECHILD,self);
    break;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint=T_SETBIT(self->element.flags,otherint,msg->otherintaux);
  case FG_SETFLAGS:
    {
    fgFlag change=self->element.flags^(fgFlag)otherint;
    self->element.flags=(fgFlag)otherint;
    if(change&FGWIN_NOCLIP) fgWindow_SetParent(self,self->element.parent); // If we change noclip its equivalent to switching the order.
    }
    break;
  case FG_SHOW: // If 0 is sent in, hide, otherwise show. Our internal flag is 1 if hidden, 0 if shown.
    self->element.flags=T_SETBIT(self->element.flags,FGWIN_HIDDEN,!msg->otherint);
    break;
  case FG_SETORDER:
    if(!msg->other2) // If it's internal, that means we need to change our order.
    {
      int old = self->element.order;
      self->element.order=msg->otherint;
      LList_ChangeOrder((fgChild*)self,&self->element.parent->root,&self->element.parent->last,FGWIN_NOCLIP);
      if(self->element.parent!=0) // Propagate up if we have a parent
      {
        FG_Msg aux={0};
        aux.type=FG_SETORDER;
        aux.otherint=old;
        aux.other2=self;
        assert(self!=0);
        return (*fgSingleton()->behaviorhook)((fgWindow*)self->element.parent,&aux);
      }
    } // Otherwise it's external, which is just a notification in case we care about it.
    break;
  case FG_MOVE:
    if(self->element.flags&(FGWIN_NOCLIP|FGWIN_HIDDEN)) // If we aren't clipping or are hidden, we NEVER propagate FG_MOVE messages.
      break;
    if(!msg->other && self->element.parent!=0) // If it's internal, we always propagate it if we have a parent.
      fgWindow_VoidAuxMessage((fgWindow*)self->element.parent,FG_MOVE,self,-(int)msg->otheraux);
    if(msg->otheraux>0) // If our dimensions changed, we have to propogate down to all our children that have nonzero relative coordinates.
    { // We check for otheraux>0 because if it's negative, it was propagated UP instead of down.
      fgChild* cur = self->element.root;
      CRect* a;
      while(hold=cur)
      {
        cur=cur->next;
        a = &hold->element.area;
        if(a->left.rel!=0 || a->top.rel!=0 || a->right.rel!=0 || a->bottom.rel!=0)
          fgWindow_VoidAuxMessage((fgWindow*)hold,FG_MOVE,self,a->left.rel!=a->right.rel || a->top.rel!=a->bottom.rel);
      }
    }
    break;
  case FG_SETSKIN:
    if(self->skin != 0)
    {
      for(FG_UINT i = 0; i < self->skinstatics.l; ++i)
        fgWindow_VoidMessage(self, FG_REMOVECHILD, fgVector_GetP(self->skinstatics, i, fgStatic*));
    }
    self->skinstatics.l = 0;
    self->skin=(struct __FG_SKIN*)msg->other;
    if(self->skin!=0)
    {
      FG_Msg aux = {0};
      aux.type=FG_ADDCHILD;
      fgFullDef* def = (fgFullDef*)self->skin->statics.p;
      for(FG_UINT i = 0; i < self->skin->statics.l; ++i)
      {
        aux.other = fgLoadDef(def[i].def, &def[i].element, def[i].order);
        (*fgSingleton()->behaviorhook)(self,&aux);
        fgVector_Add(self->skinstatics, aux.other, fgStatic*);
      }
    }
    fgWindow_TriggerStyle((fgWindow*)self, 0);
    break;
  case FG_SETSTYLE:
    if(msg->other != 0)
    {
      fgStyleMsg* cur = ((fgStyle*)msg->other)->styles;
      while(cur)
      {
        if(cur->index >= 0 && cur->index < self->skinstatics.l) // the index can be negative to represent a style that is manually applied by a message overload.
          (*fgSingleton()->behaviorhook)((fgWindow*)fgVector_Get(self->skinstatics, cur->index, fgStatic*), &cur->msg);
        cur = cur->next;
      }
    }
    break;
  case FG_DESTROY:
    VirtualFreeChild((fgChild*)self);
    break;
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgWindow";
    break;
  case FG_SETNAME:
    if(self->name) free(self->name);
    self->name = 0;
    if(msg->other)
    {
      size_t len = strlen(msg->other)+1;
      self->name = malloc(len);
      memcpy(self->name, msg->other, len);
    }
    break;
  case FG_GETNAME:
    (*(const char**)msg->other) = self->name;
    break;
  }
  return 0;
}

char FG_FASTCALL fgWindow_HoverProcess(fgWindow* self, const FG_Msg* msg)
{
  char t;
  assert(self!=0 && msg!=0);
  switch(msg->type)
  {
  case FG_MOUSEMOVE:
    if(msg->allbtn&FG_MOUSELBUTTON) //If the button is active, eat the mousemove
      return 0;
    break;
  case FG_MOUSEON:
    fgWindow_BasicMessage(self,FG_HOVER);
    break;
  case FG_MOUSEUP:
    t=MsgHitCRect(msg,&self->element);
    if(fgFocusedWindow==self && t) // Does this happen while we have focus AND the event is inside our control?
      fgWindow_BasicMessage(self,FG_CLICKED); // Fire off a message
    fgWindow_BasicMessage(self,t?FG_HOVER:FG_NUETRAL);
    break;
  case FG_MOUSEOFF: // FG_MOUSEUP falls through
    fgWindow_BasicMessage(self,FG_NUETRAL);
    break;
  case FG_MOUSEDOWN:
    if(msg->button==FG_MOUSELBUTTON && MsgHitCRect(msg,&self->element)) {
      fgWindow_BasicMessage(self,FG_ACTIVE);
    }
    break;
  }
  return fgWindow_Message(self,msg);
}

void FG_FASTCALL fgWindow_SetElement(fgWindow* self, const fgElement* element)
{
  char moved;
  fgElement* e = &self->element.element;
  moved = memcmp(e,element,sizeof(fgElement))!=0;
  if(moved)
  {
    memcpy(e,element,sizeof(fgElement));
    fgWindow_BasicMessage(self,FG_MOVE);
  }
}
void FG_FASTCALL fgWindow_SetArea(fgWindow* self, const CRect* area)
{
  char moved;
  CRect* l = &self->element.element.area;
  moved = CompareCRects(l,area);
  if(moved)
  {
    char dimchange = (l->bottom.rel-l->top.rel)!=(area->bottom.rel-area->top.rel) || 
      (l->right.rel-l->left.rel)!=(area->right.rel-area->left.rel) ||
      (l->bottom.abs-l->top.abs)!=(area->bottom.abs-area->top.abs) || 
      (l->right.abs-l->left.abs)!=(area->right.abs-area->left.abs);
    //(sseVec(&l->right.rel)-sseVec(&l->left.rel))!=(sseVec(&area->right.rel)-sseVec(&area->right.rel))
    memcpy(l,area,sizeof(CRect));
    fgWindow_VoidAuxMessage(self,FG_MOVE,0,dimchange);
  }
}

char FG_FASTCALL fgWindow_BasicMessage(fgWindow* self, unsigned char type)
{
  FG_Msg aux = {0};
  aux.type=type;
  assert(self!=0);
  return (*fgSingleton()->behaviorhook)(self,&aux);
}

char FG_FASTCALL fgWindow_VoidMessage(fgWindow* self, unsigned char type, void* data)
{
  FG_Msg aux = {0};
  aux.type=type;
  aux.other=data;
  assert(self!=0);
  return (*fgSingleton()->behaviorhook)(self,&aux);
}

char FG_FASTCALL fgWindow_VoidAuxMessage(fgWindow* self, unsigned char type, void* data, int aux)
{
  FG_Msg msg = {0};
  msg.type=type;
  msg.other=data;
  msg.otheraux=aux;
  assert(self!=0);
  return (*fgSingleton()->behaviorhook)(self,&msg);
}

char FG_FASTCALL fgWindow_IntMessage(fgWindow* self, unsigned char type, int data)
{
  FG_Msg aux = {0};
  aux.type=type;
  aux.otherint=data;
  assert(self!=0);
  return (*fgSingleton()->behaviorhook)(self,&aux);
}

void FG_FASTCALL fgWindow_SetParent(fgWindow* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent)
{
  fgChild_SetParent((fgChild*)self,parent,FGWIN_NOCLIP);
  fgWindow_BasicMessage(self,FG_MOVE);
}

void FG_FASTCALL fgWindow_TriggerStyle(fgWindow* self, FG_UINT index)
{
  if(self->skin && index < self->skin->statics.l)
    fgWindow_VoidMessage(self, FG_SETSTYLE, fgSkin_GetStyle(self->skin, index));
}

//void FG_FASTCALL DoSkinCheck(fgWindow* self, fgStatic** skins, const FG_Msg* msg)
//{
//  if(skins[msg->otheraux]!=0)
//    fgWindow_VoidMessage(self,FG_REMOVESTATIC,skins[msg->otheraux]);
//  skins[msg->otheraux]=(fgStatic*)msg->other;
//}


      /*if(active && flip!=0) // Enable clipping, remove from nonclipping array
      {
        for(i=0; i<fgNonClipping.l; ++i)
          if(fgNonClipping.p[i]==self)
          {
            Remove_VectWindow(&fgNonClipping,i);
            break;
          }
      }
      else if(!active && flip!=0) // Disable clipping, add to nonclipping array
        Add_VectWindow(&fgNonClipping,self); //DON'T sort this; if any parent window changed its zorder everything would explode
      break;*/
    /*case FGWIN_HIDDEN:
      if(active && flip!=0) // Hide window
      {
        hold=self->element.parent;
        fgWindow_VoidMessage(self,FG_SETPARENT,0);
        self->element.next=hold; // Store parent in next variable so as not to confuse SetParent or the destructor
      }
      else if(!active && flip!=0) // Show window
      {
        assert(self->element.next!=0); // next will either have the stored secret parent in it, or be NULL. You can't "Show" something that has no parent!
        fgWindow_VoidMessage(self,FG_SETPARENT,self->element.next);
      }
      break;*/