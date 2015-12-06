// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgChild.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include <math.h>
#include <limits.h>

void FG_FASTCALL fgChild_InternalSetup(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*))
{
  assert(self != 0);
  memset(self, 0, sizeof(fgChild));
  self->destroy = destroy;
  self->free = 0;
  self->message = message;
  self->flags = flags;
  self->style = (FG_UINT)-1;
  if(element) self->element = *element;
  fgChild_VoidMessage(self, FG_CONSTRUCT, 0);
  fgChild_VoidMessage(self, FG_SETPARENT, parent);
}

void FG_FASTCALL fgChild_Init(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  fgChild_InternalSetup(self, flags, parent, element, &fgChild_Destroy, &fgChild_Message);
}

void FG_FASTCALL fgChild_Destroy(fgChild* self)
{
  assert(self != 0);
  if(fgFocusedWindow == self)
    fgChild_VoidMessage(self, FG_LOSTFOCUS, 0);

  fgChild_Clear(self);
  fgChild_SetParent(self->root, 0);
  fgVector_Destroy(&self->skinrefs);
}

void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent)
{
  assert(self != 0);
  if(self->parent == parent)
    return;

  if(self->parent != 0)
  {
    fgChild_VoidMessage(self->parent, FG_LAYOUTREMOVE, self);
    if(self->parent->lastfocus == self)
      self->parent->lastfocus = 0;
    LList_RemoveAll(self); // Remove ourselves from our parent
  }
  self->next = 0;
  self->prev = 0;
  self->parent = parent;

  if(parent)
  {
    LList_AddAll(self);
    fgChild_VoidMessage(parent, FG_LAYOUTADD, self);
  }
}

char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgChild* child)
{
  CRect* area = &child->element.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  if(area->right.rel == area->left.rel && area->right.abs > d)
  {
    selfarea->right.abs = selfarea->left.abs + area->right.abs;
    return 2;
  }
  return 0;
}

char FG_FASTCALL fgLayout_ExpandY(CRect* selfarea, fgChild* child)
{
  CRect* area = &child->element.area;
  FABS d = selfarea->bottom.abs - selfarea->top.abs;
  if(area->bottom.rel == area->top.rel && area->bottom.abs > d)
  {
    selfarea->bottom.abs = selfarea->top.abs + area->bottom.abs;
    return 4;
  }
  return 0;
}

FG_FASTCALL fgLayout_ResetLayout(fgChild* self, fgChild* exclude)
{
  CRect area = self->element.area;
  fgChild* hold = self->root;
  if(self->flags & FGCHILD_EXPANDX)
    area.right.abs = area.left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
  if(self->flags & FGCHILD_EXPANDY)
    area.bottom.abs = area.top.abs;
  char dim = 0;

  while(hold)
  {
    if(hold != exclude)
    {
      if(self->flags & FGCHILD_EXPANDX)
        dim |= fgLayout_ExpandX(&area, hold);
      if(self->flags & FGCHILD_EXPANDY)
        dim |= fgLayout_ExpandY(&area, hold);
    }
    hold = hold->next;
  }

  if(dim)
    fgChild_VoidMessage(self, FG_SETAREA, &area);
}

char FG_FASTCALL fgChild_PotentialResize(fgChild* self)
{
  return ((self->element.area.left.rel != 0 || self->element.area.right.rel != 0) << 3) // If you have nonzero relative coordinates, a resize will cause a move
    | ((self->element.area.top.rel != 0 || self->element.area.bottom.rel != 0) << 4)
    | ((self->element.area.left.rel != self->element.area.right.rel) << 1) // If you have DIFFERENT relative coordinates, a resize will cause a move AND a resize for you.
    | ((self->element.area.top.rel != self->element.area.bottom.rel) << 2);
}

fgChild* FG_FASTCALL fgChild_LoadLayout(fgChild* parent, fgClassLayout* layout, fgChild* (*mapping)(const char*, fgFlag, fgChild*, fgElement*))
{
  fgChild* child = (*mapping)(layout->style.name, layout->style.flags, parent, &layout->style.element);
  fgChild_VoidAuxMessage(child, FG_SETSTYLE, 0, (ptrdiff_t)&layout->style.style);

  for(FG_UINT i = 0; i < layout->children.l; ++i)
    fgChild_LoadLayout(child, fgVector_GetP(layout->children, i, fgClassLayout), mapping);

  return child;
}

size_t FG_FASTCALL fgChild_Message(fgChild* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgChild* hold;
  assert(self != 0);
  assert(msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    return 0;
  case FG_MOVE:
    if(!msg->other && self->parent != 0) // This is internal, so we must always propagate it up
      fgChild_VoidAuxMessage(self->parent, FG_MOVE, self, msg->otheraux | 1);
    if(msg->otheraux & 1) // A child moved, so recalculate any layouts
      fgChild_VoidAuxMessage(self, FG_LAYOUTMOVE, msg->other, msg->otheraux);
    else if(msg->otheraux) // This was either internal or propagated down, in which case we must keep propagating it down so long as something changed.
    {
      fgChild* ref = !msg->other ? self : msg->other;
      fgChild* cur = self->root;
      char diff;
      while(hold = cur)
      {
        cur = cur->next;
        diff = fgChild_PotentialResize(hold);
        
        fgChild_VoidAuxMessage(hold, FG_MOVE, ref, diff & msg->otheraux);
      }

      if(msg->otheraux & 0b10000110)
        fgChild_IntMessage(self, FG_LAYOUTRESIZE, msg->otheraux, 0);
    }
    return 0;
  case FG_SETALPHA:
    return 1;
  case FG_SETAREA:
    if(!msg->other)
      return 1;
    {
      CRect* area = (CRect*)msg->other;
      char diff = CompareCRects(&self->element.area, area);
      memcpy(&self->element.area, area, sizeof(CRect));

      if(diff)
        fgChild_VoidAuxMessage(self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_SETELEMENT:
    if(!msg->other)
      return 1;
    {
      fgElement* element = (fgElement*)msg->other;
      char diff = CompareElements(&self->element, element);
      memcpy(&self->element, element, sizeof(fgElement));

      if(diff)
        fgChild_VoidAuxMessage(self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->flags, otherint, msg->otherintaux);
  case FG_SETFLAGS:
  {
    fgFlag oldflags = self->flags;
    fgFlag change = self->flags ^ (fgFlag)otherint;
    self->flags = (fgFlag)otherint;
    if(change&FGCHILD_IGNORE || change&FGCHILD_NOCLIP)
    {
      self->flags = oldflags;
      LList_RemoveAll(self);
      self->flags = (fgFlag)otherint;
      LList_AddAll(self);
    }
    if(change&FGCHILD_BACKGROUND && self->parent != 0) // If we change the background we either have to add or remove this from the layout
      fgChild_VoidMessage(self->parent, (self->flags & FGCHILD_BACKGROUND) ? FG_LAYOUTADD : FG_LAYOUTREMOVE, self);
    if((change&FGCHILD_EXPAND)&self->flags) // If we change the expansion flags, we must recalculate every single child in our layout provided one of the expansion flags is actually set
      fgLayout_ResetLayout(self, 0);
  }
    return 0;
  case FG_SETMARGIN:
    if(!msg->other)
      return 1;
    {
      AbsRect* margin = (AbsRect*)msg->other;
      char diff = CompareAbsRects(&self->margin, margin);
      memcpy(&self->margin, margin, sizeof(AbsRect));

      if(diff)
        fgChild_VoidAuxMessage(self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_SETPADDING:
    if(!msg->other)
      return 1;
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareAbsRects(&self->padding, padding);
      memcpy(&self->padding, padding, sizeof(AbsRect));

      if(diff)
        fgChild_VoidAuxMessage(self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_SETORDER:
    if(!msg->other2) // if NULL this isn't a propagation.
    {
      self->order = msg->otherint;
      if(self->parent)
      {
        fgChild* old = self->prev;
        LList_ChangeOrderAll(self);
        FG_Msg m;
        m.type = FG_SETORDER;
        m.other1 = self;
        m.other2 = old;
        fgChild_PassMessage(self->parent, &m);
      }
    }
    else
    {
      FG_Msg m = *msg;
      m.type = FG_LAYOUTREORDER;
      fgChild_PassMessage(self, &m);
    }
    return 0; // Otherwise it's just a notification
  case FG_SETPARENT:
    fgChild_SetParent(self, msg->other);
    fgChild_VoidMessage(self, FG_SETSKIN, 0); // re-evaluate our skin
    fgChild_VoidAuxMessage(msg->other, FG_MOVE, 0, fgChild_PotentialResize(self));
    return 0;
  case FG_ADDCHILD:
    hold = (fgChild*)msg->other;
    if(!hold)
      return 1;
    return fgChild_VoidMessage(hold, FG_SETPARENT, self);
  case FG_REMOVECHILD:
    hold = (fgChild*)msg->other;
    if(!msg->other || hold->parent != self)
      return 1;
    return fgChild_VoidMessage(msg->other, FG_SETPARENT, 0);
  case FG_LAYOUTRESIZE:
  case FG_LAYOUTADD:
  case FG_LAYOUTREMOVE:
  case FG_LAYOUTMOVE:
  case FG_LAYOUTREORDER:
    return fgLayout_Default(self, msg);
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = msg->other1;
    if(!layout)
      return 1;

    fgChild* (*mapping)(const char*, fgFlag, fgChild*, fgElement*) = msg->other2;
    if(!mapping) mapping = &fgLayoutLoadMapping;

    for(FG_UINT i = 0; i < layout->layout.l; ++i)
      fgChild_LoadLayout(self, fgVector_GetP(layout->layout, i, fgClassLayout), mapping);
  }
    return 0;
  case FG_CLONE:
  {
    hold = msg->other;
    if(!hold)
      hold = malloc(sizeof(fgChild));
    memcpy(hold, self, sizeof(fgChild));
    hold->root = 0;
    hold->last = 0;
    hold->parent = 0;
    hold->next = 0;
    hold->prev = 0;
    hold->nextclip = 0;
    hold->prevclip = 0;
    hold->rootnoclip = 0;
    hold->lastnoclip = 0;
    hold->rootclip = 0;
    hold->lastclip = 0;

    fgChild* cur = self->root;
    while(cur)
    {
      fgChild_VoidMessage(hold, FG_ADDCHILD, (fgChild*)fgChild_VoidMessage(cur, FG_CLONE, 0));
      cur = cur->next;
    }

    if(!msg->other)
      return (size_t)hold;
  }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgChild";
  case FG_GETSKIN:
    if(!msg->other) // If NULL this wasn't a propagation.
    {
      if(!self->parent)
        return (size_t)self->skin;
      
      int index; // First we check if we're a prechild, and if we are, check if there is a skin for us.
      for(index = 0; index < self->parent->prechild; ++index)
        if(fgVector_Get(self->parent->skinrefs, index, fgChild*) == self)
          break;
      index -= self->parent->prechild;
      if(index < 0 && self->parent->skin != 0)
      {
        for(FG_UINT i = 0; i < self->parent->skin->subskins.l; ++i)
        {
          fgSkin* skin = fgVector_GetP(self->parent->skin->subskins, index, fgSkin);
          if(skin->index == index)
            return (size_t)skin;
        }
      }

      // If not we start propagating up
      return fgChild_VoidMessage(self->parent, FG_GETSKIN, self);
    }
    else if(self->skin != 0)
    {
      hold = (fgChild*)msg->other;
      const char* name = (const char*)fgChild_VoidMessage(hold, FG_GETNAME, 0);
      if(name)
      {
        fgSkin* skin = fgSkin_GetSkin(self->skin, name);
        if(skin != 0)
          return (size_t)skin;
      }
      name = (const char*)fgChild_VoidMessage(hold, FG_GETCLASSNAME, 0);
      
      fgSkin* skin = fgSkin_GetSkin(self->skin, name);
      if(skin != 0)
        return (size_t)skin;
    }
    if(self->parent != 0)
      return fgChild_PassMessage(self->parent, msg);
    return 0;
  case FG_SETSKIN:
  {
    fgSkin* skin = (fgSkin*)(!msg->other ? (void*)fgChild_VoidMessage(self, FG_GETSKIN, 0) : msg->other);
    if(self->skin == skin) break; // no change in skin
    if(self->skin != 0)
    {
      for(FG_UINT i = self->prechild; i < self->skinrefs.l; ++i)
        fgChild_VoidMessage(self, FG_REMOVECHILD, fgVector_Get(self->skinrefs, i, fgChild*));
    }
    self->skinrefs.l = self->prechild;
    self->skin = skin;
    if(self->skin != 0)
    {
      fgChild* (*mapping)(const char*, fgFlag, fgChild*, fgElement*) = msg->other2;
      if(!mapping) mapping = &fgLayoutLoadMapping;

      for(FG_UINT i = 0; i < self->skin->children.l; ++i)
      {
        fgStyleLayout* layout = fgVector_GetP(self->skin->children, i, fgStyleLayout);
        fgChild* child = (*mapping)(layout->name, layout->flags, self, &layout->element);
        fgChild_VoidAuxMessage(child, FG_SETSTYLE, 0, (ptrdiff_t)&layout->style);
        fgVector_Add(self->skinrefs, child, fgChild*);
      }

      fgChild* cur = self->root;
      while(cur)
      {
        fgChild_VoidMessage(cur, FG_SETSKIN, 0); // This will automatically set any subskins we have if necessary.
        cur = cur->next;
      }
    }
    fgChild_IntMessage(self, FG_SETSTYLE, -1, 0);
  }
    return 0;
  case FG_SETSTYLE:
  {
    fgStyle* style = msg->other2;
    if(!style)
    {
      FG_UINT index = msg->otherint;
      if(index == -1)
        index = fgChild_VoidMessage(self, FG_GETSTYLE, 0);

      if(self->skin != 0 && index < self->skin->styles.l)
        style = fgVector_GetP(self->skin->styles, index, fgStyle);
    }

    if(style)
    {
      fgStyleMsg* cur = style->styles;
      while(cur)
      {
        fgChild_PassMessage(self, &cur->msg);
        cur = cur->next;
      }

      FG_UINT ind;
      for(FG_UINT i = 0; i < style->substyles.l; ++i)
      {
        fgStyle* substyle = fgVector_GetP(style->substyles, i, fgStyle);
        ind = substyle->index + self->prechild; // if this becomes an illegal negative number, it will wrap to an enormous positive number and fail the check below.
        if(ind < self->skinrefs.l)
        {
          hold = fgVector_Get(self->skinrefs, ind, fgChild*);
          if(hold != 0)
            fgChild_VoidAuxMessage(hold, FG_SETSTYLE, 0, (ptrdiff_t)substyle);
        }
        cur = cur->next;
      }
    }
  }
    return 0;
  case FG_GETSTYLE:
    return (self->style == (FG_UINT)-1 && self->parent != 0) ? fgChild_VoidMessage(self->parent, FG_GETSTYLE, 0) : self->style;
  case FG_GOTFOCUS:
    if(self->lastfocus) {
      fgChild* hold = self->lastfocus;
      self->lastfocus = 0;
      if(!fgChild_VoidMessage(hold, FG_GOTFOCUS, 0))
        return 0;
    }
    if(self->parent)
      return fgChild_VoidMessage(self->parent, FG_GOTFOCUS, 0);
    break;
  case FG_DRAW:
    fgStandardDraw(self, msg->other, INT_MAX);
    return 0;
  }

  return 1;
}

void FG_FASTCALL VirtualFreeChild(fgChild* self)
{
  assert(self != 0);
  (*self->destroy)(self);
  if(self->free)
    (*self->free)(self);
}

void FG_FASTCALL ResolveRect(const fgChild* self, AbsRect* out)
{
  //static const fgChild* plast=0; // This uses a simple memoization scheme so that repeated calls using the same child don't recalculate everything
  //static AbsRect last;
  AbsRect last;
  AbsVec center = { self->element.center.x.abs,self->element.center.y.abs };
  const CRect* v = &self->element.area;
  assert(out != 0);
  out->left = v->left.abs;
  out->top = v->top.abs;
  out->right = v->right.abs;
  out->bottom = v->bottom.abs;

  if(!self->parent)
    return;
  //if(plast!=self->parent)
  //{
  ResolveRect(self->parent, &last);
  //  plast=self->parent;
  //}

  out->left += lerp(last.left, last.right, v->left.rel);
  out->top += lerp(last.top, last.bottom, v->top.rel);
  out->right += lerp(last.left, last.right, v->right.rel);
  out->bottom += lerp(last.top, last.bottom, v->bottom.rel);

  center.x += (v->right.abs - v->left.abs)*self->element.center.x.rel;
  center.y += (v->bottom.abs - v->top.abs)*self->element.center.y.rel;
  //center = ResolveVec(&,r); // We can't use this because the center is relative to the DIMENSIONS, not the actual position.
  out->left -= center.x;
  out->top -= center.y;
  out->right -= center.x;
  out->bottom -= center.y;
  out->left += self->margin.left;
  out->top += self->margin.top;
  out->right -= self->margin.right;
  out->bottom -= self->margin.bottom;

  if(!(self->flags & FGCHILD_BACKGROUND))
  {
    out->left += self->padding.left;
    out->top += self->padding.top;
    out->right -= self->padding.right;
    out->bottom -= self->padding.bottom;
  }
}

void FG_FASTCALL ResolveRectCache(AbsRect* BSS_RESTRICT out, const fgChild* elem, const AbsRect* BSS_RESTRICT last)
{
  AbsVec center = { elem->element.center.x.abs, elem->element.center.y.abs };
  const CRect* v = &elem->element.area;
  assert(out != 0 && elem != 0 && last != 0);
  out->left = lerp(last->left, last->right, v->left.rel) + v->left.abs;
  out->top = lerp(last->top, last->bottom, v->top.rel) + v->top.abs;
  out->right = lerp(last->left, last->right, v->right.rel) + v->right.abs;
  out->bottom = lerp(last->top, last->bottom, v->bottom.rel) + v->bottom.abs;

  center.x += (v->right.abs - v->left.abs)*elem->element.center.x.rel;
  center.y += (v->bottom.abs - v->top.abs)*elem->element.center.y.rel;
  //center = ResolveVec(&,r); // We can't use this because the center is relative to the DIMENSIONS, not the actual position.
  out->left -= center.x;
  out->top -= center.y;
  out->right -= center.x;
  out->bottom -= center.y;
  out->left += elem->margin.left;
  out->top += elem->margin.top;
  out->right -= elem->margin.right;
  out->bottom -= elem->margin.bottom;

  if(!(elem->flags & FGCHILD_BACKGROUND))
  {
    out->left += elem->padding.left;
    out->top += elem->padding.top;
    out->right -= elem->padding.right;
    out->bottom -= elem->padding.bottom;
  }
}


char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgChild* child)
{
  AbsRect r;
  assert(msg != 0 && child != 0);
  ResolveRect(child, &r);
  return MsgHitAbsRect(msg, &r);
}

void FG_FASTCALL LList_RemoveAll(fgChild* self)
{
  LList_Remove(self, &self->parent->root, &self->parent->last); // Remove ourselves from our parent
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_Remove(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Remove(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}
void FG_FASTCALL LList_Remove(fgChild* self, fgChild** root, fgChild** last)
{
  assert(self != 0);
  if(self->prev != 0) self->prev->next = self->next;
  else *root = self->next;
  if(self->next != 0) self->next->prev = self->prev;
  else *last = self->prev;
}

void FG_FASTCALL LList_Insert(fgChild* self, fgChild* cur, fgChild* prev, fgChild** root, fgChild** last)
{
  self->next = cur;
  self->prev = prev;
  if(prev) prev->next = self;
  else *root = self; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev = self;
  else *last = self; // Cur is null if we are at the end of the list, so update last
}

void FG_FASTCALL LList_ChangeOrderAll(fgChild* self)
{
  LList_ChangeOrder(self, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_ChangeOrder(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_ChangeOrder(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}

void FG_FASTCALL LList_ChangeOrder(fgChild* self, fgChild** root, fgChild** last)
{
  fgChild* cur = self->next;
  fgChild* prev = self->prev;
  while(cur != 0 && (self->order < cur->order))
  {
    prev = cur;
    cur = cur->next;
  }
  while(prev != 0 && (self->order > prev->order))
  {
    cur = prev;
    prev = prev->prev;
  }

  if(cur == self->next) { assert(prev == self->prev); return; } // we didn't move anywhere
  LList_Remove(self, root, last);
  LList_Insert(self, cur, prev, root, last);
}

void FG_FASTCALL LList_AddAll(fgChild* self)
{
  LList_Add(self, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_Add(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Add(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}

void FG_FASTCALL LList_Add(fgChild* self, fgChild** root, fgChild** last)
{
  fgChild* cur = *root;
  fgChild* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(self != 0 && root != 0);
  if(!cur) // We do this check up here because we'd have to do it for the end append check below anyway so we might as well utilize it!
    LList_Insert(self, 0, 0, root, last);
  else if(self->order < (*last)->order) // shortcut for appending to the end of the list
    LList_Insert(self, 0, *last, root, last);
  else
  {
    while(cur != 0 && (self->order < cur->order))
    {
      prev = cur;
      cur = cur->next;
    }
    LList_Insert(self, cur, prev, root, last);
  }
}

size_t FG_FASTCALL fgChild_VoidMessage(fgChild* self, unsigned char type, void* data)
{
  FG_Msg aux = { 0 };
  aux.type = type;
  aux.other = data;
  assert(self != 0);
  return (*fgSingleton()->behaviorhook)(self, &aux);
}

size_t FG_FASTCALL fgChild_VoidAuxMessage(fgChild* self, unsigned char type, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgSingleton()->behaviorhook)(self, &msg);
}

size_t FG_FASTCALL fgChild_IntMessage(fgChild* self, unsigned char type, ptrdiff_t data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.otherint = data;
  msg.otherintaux = aux;
  assert(self != 0);
  return (*fgSingleton()->behaviorhook)(self, &msg);
}

FG_EXTERN size_t FG_FASTCALL fgChild_PassMessage(fgChild* self, const FG_Msg* msg)
{
  return (*fgSingleton()->behaviorhook)(self, msg);
}

size_t FG_FASTCALL fgLayout_Default(fgChild* self, const FG_Msg* msg)
{
  if(!(self->flags & FGCHILD_EXPAND))
    return 0;

  fgChild* child = msg->other;

  switch(msg->type)
  {
  case FG_LAYOUTREORDER:
  case FG_LAYOUTRESIZE:
    break;
  case FG_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.type = FG_LAYOUTREMOVE;
    fgLayout_Default(self, &m);
    m.type = FG_LAYOUTADD;
    fgLayout_Default(self, &m);
    break;
  }
  case FG_LAYOUTADD:
  {
    if(child->flags & FGCHILD_BACKGROUND)
      break;
    CRect area = self->element.area;
    char dim = 0;
    if(self->flags & FGCHILD_EXPANDX)
      dim |= fgLayout_ExpandX(&area, child);
    if(self->flags & FGCHILD_EXPANDY)
      dim |= fgLayout_ExpandY(&area, child);
    if(dim)
      fgChild_VoidMessage(self, FG_SETAREA, &area);
    break;
  }
  case FG_LAYOUTREMOVE:
  {
    if(child->flags & FGCHILD_BACKGROUND)
      break;
    CRect* childarea = &child->element.area;
    FABS dx = self->element.area.right.abs - self->element.area.left.abs;
    FABS dy = self->element.area.bottom.abs - self->element.area.top.abs;
    if(((self->flags & FGCHILD_EXPANDX) &&
        childarea->right.rel == childarea->left.rel &&
        childarea->right.abs >= dx) ||
      ((self->flags & FGCHILD_EXPANDY) &&
        childarea->bottom.rel == childarea->top.rel &&
        childarea->bottom.abs >= dy))
      fgLayout_ResetLayout(self, child);
  }
    break;
  }

  return 0;
}

char fgLayout_TileSame(FABS posx, FABS posy, fgChild* cur)
{
  return (posx > 0 || posy > 0) && posx == cur->element.area.left.abs && posy == cur->element.area.left.abs;
}

void fgLayout_TileReorder(fgChild* prev, fgChild* child, char axis, FABS max, FABS pitch) // axis: 0 means x-axis first, 1 means y-axis first
{
  AbsVec pos = { 0,0 };
  if(prev)
  {
    pos.x = axis ? prev->element.area.bottom.abs : prev->element.area.right.abs;
    pos.y = axis ? prev->element.area.left.abs : prev->element.area.top.abs;
  }
  fgChild* cur = child;
  char rowbump = 0; // Used to keep track if we've gone down at least one row - this is required for our same position optimization to be valid.

  while(cur)
  {
    if(!(cur->flags&FGCHILD_BACKGROUND))
    {
      if(fgLayout_TileSame(axis ? pos.y : pos.x, axis ? pos.x : pos.y, cur) && rowbump)
        break;
      AbsVec dim = { prev->element.area.right.abs - prev->element.area.left.abs, prev->element.area.bottom.abs - prev->element.area.top.abs };
      if(axis) { FABS t = dim.x; dim.x = dim.y; dim.y = t; }

      if(pos.x + dim.x > max)
      {
        pos.y += pitch;
        pos.x = 0;
        pitch = 0;
        rowbump = 1;
      }

      if(!axis)
        MoveCRect(pos, &cur->element.area);
      else
        MoveCRectInv(pos, &cur->element.area);
      pos.x += dim.x;
      if(dim.y > pitch) pitch = dim.y;
    }
    cur = cur->next;
  }
}

FABS FG_FASTCALL fgLayout_TileGetPitch(fgChild* cur, char axis)
{
  FABS start = (axis ? cur->element.area.left.abs : cur->element.area.top.abs);
  FABS dim;
  FABS pitch = 0.0;
  while((axis ? cur->element.area.left.abs : cur->element.area.top.abs) == start)
  {
    dim = (axis ? cur->element.area.right.abs : cur->element.area.bottom.abs) - start;
    if(dim > pitch)
      pitch = dim;
    cur = cur->prev; // run backwards until we go up a level
  }
  return pitch;
}

size_t FG_FASTCALL fgLayout_Tile(fgChild* self, const FG_Msg* msg, char axes)
{
  fgChild* child = (fgChild*)msg->other;
  char axis = ((axes & 3) && (axes & 8)) || ((axes & 2) && !(axes & 1));
  FABS max = INFINITY;
  if(axes & 3)
  {
    AbsRect out;
    ResolveRect(self, &out);
    if((axes & 8) && !(self->flags & FGCHILD_EXPANDY)) max = out.bottom - out.top;
    else if(!(self->flags & FGCHILD_EXPANDX)) max = out.right - out.left;
  }

  switch(msg->type)
  {
  case FG_LAYOUTMOVE:
    break;
  case FG_LAYOUTRESIZE:
    if((((msg->otherint&0b10) && !(self->flags&FGCHILD_EXPANDX)) || ((msg->otherint & 0b100) && !(self->flags&FGCHILD_EXPANDY))) && (axes & 3))
      fgLayout_TileReorder(0, self->root, axis, max, 0.0f);
    break;
  case FG_LAYOUTREORDER:
  {
    fgChild* old = (fgChild*)msg->other2;
    child = child->order < old->order ? child : old; // Get lowest child
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
  }
    break;
  case FG_LAYOUTADD:
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  case FG_LAYOUTREMOVE:
    fgLayout_TileReorder(child->prev, child->next, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Distribute(fgChild* self, const FG_Msg* msg, char axis)
{
  /*switch(msg->type)
  {
  case FG_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FG_LAYOUTRESIZE:
    break;
  case FG_LAYOUTREORDER:
    //child = child->order<msg->other->order?child:msg->other; // Get lowest child
    fgLayout_DistributeReorder(child->prev, child, axis, num);
    break;
  case FG_LAYOUTADD:
    fgLayout_DistributeReorder(child->prev, child, axis, num);
    break;
  case FG_LAYOUTREMOVE:
    fgLayout_DistributeReorder(child->prev, child->next, axis, num);
    break;
  }*/

  return 1;
}

void FG_FASTCALL fgChild_Clear(fgChild* self)
{
  while(self->root) // Destroy all children
    VirtualFreeChild(self->root);
}

void FG_FASTCALL fgChild_AddPreChild(fgChild* self, fgChild* child)
{
  fgVector_Insert(self->skinrefs, child, self->prechild, fgChild*);
  ++self->prechild;
}

FG_EXTERN char* FG_FASTCALL fgCopyText(const char* text)
{
  if(!text) return 0;
  size_t len = strlen(text) + 1;
  char* ret = malloc(len);
  memcpy(ret, text, len);
  return ret;
}
