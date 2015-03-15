// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgGrid.h"
#include "bss_defines.h"

void FG_FASTCALL fgGrid_Init(fgGrid* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  assert(self!=0);
  fgWindow_Init(&self->window,parent,element,id,flags);
  self->window.message=&fgGrid_Message;
  fgVector_Init(&self->items);
  fgWindow_IntMessage(&self->window,FG_SETCOLUMNS,1);
}
void FG_FASTCALL fgGrid_Destroy(fgGrid* self)
{
  fgVector_Destroy(&self->items); // Don't need to destroy windows because they're children of the grid and will get blown up by its destructor
  fgWindow_Destroy(&self->window);
}

/*void FG_FASTCALL fgGrid_FixUp(fgGrid* self, int start) //
{
  int i = self->items.l;
  fgChild** items = (fgChild**)self->items.p;
  fgChild** root = &self->window.element.root;
  fgChild** last = &self->window.element.last;
  fgChild** sroot = (fgChild**)&self->window.rlist;
  fgChild** slast = (fgChild**)&self->window.rlast;

  while(--i >= start)
  {
    items[i]->order=i;
    if(items[i]->flags&FGSTATIC_MARKER)
      LList_ChangeOrder(items[i],root,last,0);
    else 
      LList_ChangeOrder(items[i],sroot,slast,FGWIN_NOCLIP);
  }
}*/
/*char FG_FASTCALL fgGridInternal_Message(fgWindow* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOVE:
    if(msg->other!=0 && msg->otheraux>=0) // If this is true it was propagated up from a child, so propagate up to our parent
      return (*((fgWindow*)self->element.parent)->message)(self->element.parent,msg); // Bypass behavior hooks here because this is internal
    break; // Otherwise let the default handler deal with it
  case FG_SETORDER:
    if(msg->other2!=0)
      return (*((fgWindow*)self->element.parent)->message)(self->element.parent,msg);
    break;
  case FG_REMOVECHILD:
    return (*((fgWindow*)self->element.parent)->message)(self->element.parent,msg);
  }

  return fgWindow_Message(self,msg);
}*/
char fgGrid_RemoveItem(fgGrid* self, char* skip, FG_Msg* aux)
{
  int i;
  aux->type=FG_REMOVECHILD;
  fgChild* hold=(fgChild*)aux->other;
  i=hold->order;
  char b = hold->element.area.bottom.abs>(self->window.element.element.area.bottom.abs-1) || hold->element.area.right.abs>(self->window.element.element.area.right.abs-1);
  hold=hold->next; // First we save the image afterwards
  if(!fgWindow_Message((fgWindow*)self, &aux)) // Then we have the window remove the image
  {
    skip=1;
    fgVector_Remove(&self->items, i, sizeof(fgChild*));
    //fgGrid_FixUp(self,hold->order);
    // When not tiling, if one of the removed windows borders are exactly equal to the bounds, then we have to re-calculate everything
    if((self->window.element.flags&(FGGRID_TILEX|FGGRID_TILEY))==0 && b)
      fgGrid_Reposition(self, 0);
    else if(hold!=0) // Otherwise we just reposition the image afterwards, if there is one, which will fix the grid.
      fgGrid_Reposition(self, hold);
    skip=0;
    return 0;
  }
  return 1;
}
char FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg)
{
  static char skip=0;
  char b;
  int k;
  fgChild* hold;
  assert(self!=0 && msg!=0);

  switch(msg->type)
  {
  case FG_MOVE:
    hold=(fgChild*)msg->other;
    if(hold!=0) { // If its external we do a reposition, which will then send an internal move message that will propagate
      if(skip) return 1;
      if(hold->order<0 || hold->order>self->items.l || hold!=fgVector_Get(self->items, hold->order, fgChild*)) return 1; // If false, this isn't actually an item
      skip=1;
      if(msg->otheraux<0) // This is a size propagation coming down from our parent, so we have to re-do everything.
        fgGrid_Reposition(self,self->window.element.root);
      else // Otherwise it's coming up from a child, so reposition only it. Do this even if it's dimensions didn't change, because it could have been added.
        fgGrid_Reposition(self,hold);
      skip=0;
    }
    break; // Otherwise let the default handler deal with it
  case FG_SETORDER:
    hold=(fgChild*)msg->other2;
    if(hold!=0) // If other is non-null, it means a child changed their order, so check it
    {
      fgChild* next=hold->next;
      if(msg->otherint<0 || msg->otherint>self->items.l || hold!=fgVector_Get(self->items, msg->otherint, fgChild*)) return 0; // If false, this isn't actually an item, so we don't care.
      fgVector_Remove(&self->items,msg->otherint,sizeof(fgChild*)); // Otherwise, this IS an item, so we need to move it to it's new location.
      if(hold->order<0 || (FG_UINT)hold->order>self->items.l) hold->order=self->items.l;
      fgVector_Insert(self->items,hold,hold->order,fgChild*); // And re-insert into it's new location.
      //hold=(!next || hold->order<next->order)?hold:next; // and fix the lowest window, either the re-ordered one, or the one that came after it before it was re-ordered.
      //fgGrid_FixUp(self,hold->order);
      fgGrid_Reposition(self,hold); 
    }
    break;
  case FG_ADDITEM:
    hold=(fgChild*)msg->other;
    k=msg->otheraux;
    if(k<0 || (FG_UINT)k>self->items.l) k=self->items.l;
    fgVector_Insert(self->items, hold, k, fgChild*);
    //fgGrid_FixUp(self,k+1);
    hold->order=0;
    b = fgWindow_VoidMessage((fgWindow*)self,FG_ADDCHILD,msg->other); // Have the window insert the item. This will trigger an FG_MOVE that gets propagated up
    hold->order=msg->otheraux;
    return b;
  case FG_GETITEM:
    if(self->items.l<=msg->otheraux)
      return 1;
    *((fgChild**)msg->other) = fgVector_Get(self->items, msg->otheraux, fgChild*);
    return 0;
  case FG_REMOVEITEM:
    if(self->items.l<=msg->otherint)
      return 1;
    {
      FG_Msg aux ={ { (void*)fgVector_Get(self->items, msg->otherint, fgChild*), (int)0 }, FG_REMOVECHILD };
      return fgGrid_RemoveItem(self, &skip, &aux);
    }
  case FG_REMOVECHILD:
    {
      FG_Msg aux = *msg;
      aux.type=FG_REMOVECHILD;
      return fgGrid_RemoveItem(self, &skip, &aux);
    }
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgGrid";
    return 0;
  }
  return fgWindow_Message((fgWindow*)self,msg);
}
FG_EXTERN void FG_FASTCALL fgGrid_Reposition(fgGrid* self, fgChild* p)
{
  CRect* hold=&self->window.element.element.area;
  AbsVec dim = { hold->right.abs-hold->left.abs, hold->bottom.abs-hold->top.abs }; // this is NOT the resolved rect because it's only used when we're expanding
  AbsRect area;
  CRect narea;
  fgFlag flags = self->window.element.flags;
  AbsVec cur = { 0,0 };
  fgChild** items = (fgChild**)self->items.p;
  FG_UINT i = p->order;

  if((flags&FGGRID_TILEX) && (flags&FGGRID_TILEY)) {
    ResolveRect((fgChild*)self,&area);
    if(self->window.element.flags&FGGRID_EXPANDX) { // If we're expanding on the X axis we must grow downward and then loop around.
      if(i>0)
      {
        //i is decremented FIRST so that if we're repositioning something on the beginning of a row, we recheck everything on the previous row.
        cur.x=items[i-1]->element.area.left.abs;
        cur.y=items[i-1]->element.area.bottom.abs;
      }
      for(; i < self->items.l; ++i)
      {
        hold=&items[i]->element.area;
        if(hold->bottom.abs-hold->top.abs+cur.y>area.bottom)
        {
          if(cur.y>dim.y) dim.y=cur.y;
          cur.y=0;
          cur.x=dim.x;
        }
        MoveCRect(cur,hold);
        cur.y=hold->bottom.abs;
        if(hold->right.abs>dim.x) dim.x=hold->right.abs;
      }
      if(cur.y>dim.y) dim.y=cur.y;
    } else { // EXPANDY is the default even if its not specified. If we expand on the Y axis, we must grow rightward.
      if(i>0)
      {
        cur.x=items[i-1]->element.area.right.abs;
        cur.y=items[i-1]->element.area.top.abs;
      }
      for(; i < self->items.l; ++i)
      {
        hold=&items[i]->element.area;
        if(hold->right.abs-hold->left.abs+cur.x>area.right)
        {
          if(cur.x>dim.x) dim.x=cur.x;
          cur.x=0;
          cur.y=dim.y;
        }
        MoveCRect(cur,hold);
        cur.x=hold->right.abs;
        if(hold->bottom.abs>dim.y) dim.y=hold->bottom.abs;
      }
      if(cur.x>dim.x) dim.x=cur.x;
    }
  } else if(flags&FGGRID_TILEX) {
    if(i>0) cur.x=items[i-1]->element.area.right.abs;
    for(; i < self->items.l; ++i)
    {
      hold=&items[i]->element.area;
      MoveCRect(cur,hold);
      cur.x=hold->right.abs;
      if(hold->bottom.abs>dim.y) dim.y=hold->bottom.abs;
      p=p->next;
    }
    dim.x=cur.x;
  } else if(flags&FGGRID_TILEY) {
    if(i>0) cur.y=items[i-1]->element.area.bottom.abs;
    for(; i < self->items.l; ++i)
    {
      MoveCRect(cur,&items[i]->element.area);
      cur.y=p->element.area.bottom.abs;
      if(p->element.area.right.abs>dim.x) dim.x=p->element.area.right.abs;
    }
    dim.y=cur.y;
  } else { // No grid flags are set, so we don't reposition any elements, we only grow in size.
    if(!p) // If null, do ALL of them because we removed something on the border
    {
      for(i=0; i < self->items.l; ++i)
      {
        hold=&items[i]->element.area;
        if(hold->right.rel==0.0 && hold->right.abs>dim.x) dim.x=hold->right.abs;
        if(hold->bottom.rel==0.0 && hold->bottom.abs>dim.y) dim.y=hold->bottom.abs;
      }
    }
    else // If not null we're inserting something so just check that one.
    {
      hold=&p->element.area;
      if(hold->right.rel==0.0 && hold->right.abs>dim.x) dim.x=hold->right.abs;
      if(hold->bottom.rel==0.0 && hold->bottom.abs>dim.y) dim.y=hold->bottom.abs;
    }
  }
  
  narea = self->window.element.element.area; // at this point, dim is always equal to the total area regardless of whether we wanted it.
  if(flags&FGGRID_EXPANDX) narea.right.abs=narea.left.abs+dim.x;
  if(flags&FGGRID_EXPANDY) narea.bottom.abs=narea.top.abs+dim.y;
  fgWindow_SetArea((fgWindow*)self,&narea); // This sends an FG_MOVE message to ourselves, which will propagate upwards when appropriate.
}

/*
fgChild* FG_FASTCALL fgGrid_HitElement(fgGrid* self, FABS x, FABS y)
{
  AbsRect dim;
  //AbsRect chdim;
  //char lasthit=0;
  fgChild* cur=self->window.element.root;
  if(!cur) cur=(fgChild*)self->window.rlist;
  ResolveRect((fgChild*)self,&dim);

  // This doesn't work
  if((self->window.element.flags&FGGRID_TILEX)!=0 == (self->window.element.flags&FGGRID_TILEY)!=0) { // Either TILEX and TILEY are both on or both off
    while(cur!=0 && (
      (x >= dim.right+cur->element.area.right.abs) || 
      (x < dim.left+cur->element.area.left.abs) || 
      (y >= dim.bottom+cur->element.area.bottom.abs) || 
      (y < dim.top+cur->element.area.top.abs))) 
      cur=cur->next;
    //while(cur!=0 && !lasthit)
    //{
    //  ResolveRectCache(&chdim,(fgElement*)cur,&dim);
    //  lasthit=HitAbsRect(&chdim,x,y);
    //  cur=cur->next;
    //}
  } else if((self->window.element.flags&FGGRID_TILEX)!=0) { // Only TILEX is active
    while(cur!=0 && ((x >= dim.right+cur->element.area.right.abs) || (x < dim.left+cur->element.area.left.abs)))
      cur=cur->next;
  } else { // only TILEY is active
    while(cur!=0 && ((y >= dim.bottom+cur->element.area.bottom.abs) || (y < dim.top+cur->element.area.top.abs)))
      cur=cur->next;
  }
  return cur;
}*/
