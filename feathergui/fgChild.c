// Copyright ©2013 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgChild.h"
#include "fgRoot.h"
#include "fgLayout.h"

void FG_FASTCALL fgChild_Init(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  assert(self != 0);
  memset(self, 0, sizeof(fgChild));
  self->destroy = &fgChild_Destroy;
  self->free = &free;
  self->message = &fgChild_Message;
  self->flags = flags;
  fgChild_SetParent(self, parent);
  if(element) self->element = *element;
}
void FG_FASTCALL fgChild_Destroy(fgChild* self)
{
  assert(self != 0);
  fgChild_Clear(self);
  fgChild_SetParent(self->root, 0);
  fgVector_Destroy(&self->skinrefs);
}

void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent)
{
  assert(self != 0);
  if(self->parent == parent) { // If this is true, we just need to make sure the ordering is valid
    if(parent)
      LList_ChangeOrder(self, &parent->root, &parent->last);
    return;
  }

  if(self->parent != 0)
  {
    fgChild_VoidMessage(self->parent, FG_LAYOUTREMOVE, self);
    LList_Remove(self, &self->parent->root, &self->parent->last); // Remove ourselves from our parent
  }
  self->next = 0;
  self->prev = 0;
  self->parent = parent;

  if(parent)
  {
    LList_Add(self, &parent->root, &parent->last);
    fgChild_VoidMessage(parent, FG_LAYOUTADD, self);
  }
}

char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgChild* child)
{
  CRect* area = &child->element.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  if(area->right.rel == 0.0 && area->right.abs > d)
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
  if(area->bottom.rel == 0.0 && area->bottom.abs > d)
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
  fgChild* child = (*mapping)(layout->name, layout->flags, parent, &layout->element);

  for(FG_UINT i = 0; i < layout->statics.l; ++i)
  {
    fgFullDef* def = fgVector_GetP(layout->statics, i, fgFullDef);
    fgChild_VoidMessage(child, FG_ADDCHILD, fgLoadDef(def->def, &def->element, def->order));
  }

  fgChild_VoidMessage(child, FG_SETSTYLE, &layout->style);

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
  case FG_SETALPHA:
    break;
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
    break;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->flags, otherint, msg->otherintaux);
  case FG_SETFLAGS:
  {
    fgFlag change = self->flags ^ (fgFlag)otherint;
    self->flags = (fgFlag)otherint;
    if(change&FGCHILD_NOCLIP) // If we change noclip its equivalent to switching the order.
      fgChild_SetParent(self, self->parent); 
    if(change&FGCHILD_BACKGROUND && self->parent != 0) // If we change the background we either have to add or remove this from the layout
      fgChild_VoidMessage(self->parent, (self->flags & FGCHILD_BACKGROUND) ? FG_LAYOUTADD : FG_LAYOUTREMOVE, self);
    if((change&(FGCHILD_EXPANDX | FGCHILD_EXPANDY))&self->flags) // If we change the expansion flags, we must recalculate every single child in our layout provided one of the expansion flags is actually set
      fgLayout_ResetLayout(self, 0);
  }
  break;
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
    break;
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
    break;
  case FG_SETORDER:
    if(!msg->other2) // if NULL this isn't a propagation.
    {
      int old = self->order;
      self->order = msg->otherint;
      if(self->parent)
      {
        LList_ChangeOrder(self, &self->parent->root, &self->parent->last);
        FG_Msg m;
        m.type = FG_SETORDER;
        m.otherint = old;
        m.other2 = self;
        (*fgSingleton()->behaviorhook)(self->parent, &m);
      }
    }
    else
    {
      FG_Msg m = *msg;
      m.type = FG_LAYOUTREORDER;
      (*fgSingleton()->behaviorhook)(self, &m);
    }
    break; // Otherwise it's just a notification
  case FG_SETPARENT:
    fgChild_SetParent(self, msg->other);
    fgChild_VoidAuxMessage(msg->other, FG_MOVE, 0, fgChild_PotentialResize(self));
    break;
  case FG_ADDCHILD:
    hold = (fgChild*)msg->other;
    if(!hold)
      return 1;
    fgChild_VoidMessage(hold, FG_SETPARENT, self);
    break;
  case FG_REMOVECHILD:
    hold = (fgChild*)msg->other;
    if(!msg->other || hold->parent != self)
      return 1;
    fgChild_VoidMessage(msg->other, FG_SETPARENT, 0);
    break;
  case FG_LAYOUTRESIZE:
  case FG_LAYOUTADD:
  case FG_LAYOUTREMOVE:
  case FG_LAYOUTMOVE:
  case FG_LAYOUTREORDER:
    fgLayout_Default(self, msg);
    break;
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
    break;
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

    fgChild* cur = self->root;
    while(cur)
    {
      fgChild_VoidMessage(hold, FG_ADDCHILD, (fgChild*)fgChild_VoidMessage(cur, FG_CLONE, 0));
      cur = cur->next;
    }

    if(!msg->other)
      return (size_t)hold;
  }
    break;
  case FG_SETFONTCOLOR:
  case FG_SETFONT: // propagate this to all children
    hold = self->root;
    while(hold)
    {
      (*fgSingleton()->behaviorhook)(hold, msg);
      hold = hold->next;
    }
  case FG_GETCLASSNAME:
    (*(const char**)msg->other) = "fgChild";
    return 0;
  case FG_SETSKIN:
    if(self->skin != 0)
    {
      for(FG_UINT i = self->prechild; i < self->skinrefs.l; ++i)
        fgChild_VoidMessage(self, FG_REMOVECHILD, fgVector_Get(self->skinrefs, i, fgChild*));
    }
    self->skinrefs.l = self->prechild;
    self->skin = (struct __FG_SKIN*)msg->other;
    if(self->skin != 0)
    {
      FG_Msg aux = { 0 };
      aux.type = FG_ADDCHILD;
      fgFullDef* def = (fgFullDef*)self->skin->defs.p;
      for(FG_UINT i = 0; i < self->skin->defs.l; ++i)
      {
        aux.other = fgLoadDef(def[i].def, &def[i].element, def[i].order);
        (*fgSingleton()->behaviorhook)(self, &aux);
        fgVector_Add(self->skinrefs, aux.other, fgChild*);
      }

      fgSkin* subskins = (fgSkin*)self->skin->subskins.p;
      for(FG_UINT i = 0; i < self->skin->subskins.l; ++i)
      {
        FG_UINT ind = subskins[i].index + self->prechild;
        if(ind < self->skinrefs.l)
          fgChild_VoidMessage(fgVector_Get(self->skinrefs, ind, fgChild*), FG_SETSKIN, subskins + i);
      }
    }
    fgChild_TriggerStyle(self, 0);
    break;
  case FG_SETSTYLE:
    if(msg->other != 0)
    {
      fgStyle* style = (fgStyle*)msg->other;
      fgStyleMsg* cur = style->styles;
      while(cur)
      {
        (*fgSingleton()->behaviorhook)(self, &cur->msg);
        cur = cur->next;
      }

      FG_UINT ind;
      for(FG_UINT i = 0; i < style->substyles.l; ++i)
      {
        fgStyle* substyle = fgVector_GetP(style->substyles, i, fgStyle);
        ind = substyle->index + self->prechild; // if this becomes an illegal negative number, it will wrap to an enormous positive number and fail the check below.
        if(ind < self->skinrefs.l)
          fgChild_VoidMessage(fgVector_Get(self->skinrefs, ind, fgChild*), FG_SETSTYLE, substyle);
        cur = cur->next;
      }
    }
    break;
  }

  return 0;
}

void FG_FASTCALL VirtualFreeChild(fgChild* self)
{
  assert(self != 0);
  (*self->destroy)(self);
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

void FG_FASTCALL LList_Remove(fgChild* self, fgChild** root, fgChild** last)
{
  assert(self != 0);
  if(self->prev != 0) self->prev->next = self->next;
  else *root = self->next;
  if(self->next != 0) self->next->prev = self->prev;
  else *last = self->prev;
}

void FG_FASTCALL LList_ChangeOrder(fgChild* self, fgChild** root, fgChild** last)
{
  static const fgFlag flag = FGCHILD_NOCLIP;
  fgChild* cur = self->next;
  fgChild* prev = self->prev;
  while(cur != 0 && (((self->flags&flag) < (cur->flags&flag)) || ((self->order < cur->order) && (self->flags&flag) == (cur->flags&flag))))
  {
    prev = cur;
    cur = cur->next;
  }
  while(prev != 0 && !(((self->flags&flag) < (prev->flags&flag)) || ((self->order < prev->order) && (self->flags&flag) == (prev->flags&flag))))
  {
    cur = prev;
    prev = prev->prev;
  }

  if(cur == self->next) { assert(prev == self->prev); return; } // we didn't move anywhere
  LList_Remove(self, root, last);
  self->next = cur;
  self->prev = prev;
  if(prev) prev->next = self;
  else *root = self; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev = self;
  else *last = self; // Cur is null if we are at the end of the list, so update last
}

//me cur
//y  y  | order<order
//y  n  | false
//n  y  | true
//n  n  | order<order

void FG_FASTCALL LList_Add(fgChild* self, fgChild** root, fgChild** last)
{
  static const fgFlag flag = FGCHILD_NOCLIP;
  fgChild* cur = *root;
  fgChild* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(self != 0 && root != 0);
  if(!cur) // We do this check up here because we'd have to do it for the end append check below anyway so we might as well utilize it!
    (*last) = (*root) = self;
  //else if(self->order<(*last)->order) { // shortcut for appending to the end of the list
  else if(((self->flags&flag) < ((*last)->flags&flag)) || ((self->order < (*last)->order) && (self->flags&flag) == ((*last)->flags&flag))) {
    cur = 0;
    prev = *last;
  }
  else {
    while(cur != 0 && (((self->flags&flag) < (cur->flags&flag)) || ((self->order < cur->order) && (self->flags&flag) == (cur->flags&flag))))
    {
      prev = cur;
      cur = cur->next;
    }
  }
  self->next = cur;
  self->prev = prev;
  if(prev) prev->next = self;
  else *root = self; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) cur->prev = self;
  else *last = self; // Cur is null if we are at the end of the list, so update last
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

void FG_FASTCALL fgChild_TriggerStyle(fgChild* self, FG_UINT index)
{
  if(self->skin && index < self->skin->styles.l)
    fgChild_VoidMessage(self, FG_SETSTYLE, fgSkin_GetStyle(self->skin, index));
}

size_t FG_FASTCALL fgLayout_Default(fgChild* self, const FG_Msg* msg)
{
  if(!(self->flags & (FGCHILD_EXPANDX | FGCHILD_EXPANDY)))
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
    if(((self->flags & FGCHILD_EXPANDX) && childarea->right.rel == 0.0 && childarea->right.abs > dx) || ((self->flags & FGCHILD_EXPANDY) && childarea->bottom.rel == 0.0 && childarea->bottom.abs > dy))
      fgLayout_ResetLayout(self, child);
  }
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Tile(fgChild* self, const FG_Msg* msg, char axes)
{
  switch(msg->type)
  {
  case FG_LAYOUTMOVE:
    break;
  case FG_LAYOUTRESIZE:
    if((self->flags&FGCHILD_EXPANDX) != 0 && (self->flags&FGCHILD_EXPANDY) != 0)
    {

    }
    break;
  case FG_LAYOUTREORDER:
    break;
  case FG_LAYOUTADD:
  case FG_LAYOUTREMOVE:
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Distribute(fgChild* self, const FG_Msg* msg, char axis)
{
  return 0;
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
