// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBox.h"
#include "fgRoot.h"
#include "bss-util/algo.h"
#include "feathercpp.h"

void fgBox_Init(fgBox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgBox_Destroy, (fgMessage)&fgBox_Message);
}
void fgBox_Destroy(fgBox* self)
{
  ((fgElementArray&)self->selected).~ArraySort();
  fgBoxOrderedElement_Destroy(&self->order);
  self->scroll->message = (fgMessage)fgScrollbar_Message;
  fgScrollbar_Destroy(&self->scroll); // this will destroy our prechildren for us.
}
void fgBoxOrderedElement_Destroy(struct _FG_BOX_ORDERED_ELEMENTS_* self)
{
  ((bss::DynArray<fgElement*>&)self->ordered).~DynArray();
}

bool BSS_FORCEINLINE checkIsOrdered(fgElement* root)
{
  if(!root) return true;
  char count = !(root->flags&FGELEMENT_BACKGROUND);
  root = root->next;
  while(count < 3 && root != 0)
  {
    if(!(root->flags&FGELEMENT_BACKGROUND) == !(count % 2))
      ++count;
    root = root->next;
  }
  return count < 3; // If count never exceeded 2, then this matches the pattern BACKGROUND-FOREGROUND-BACKGROUND.
}

template<fgFlag FLAGS> BSS_FORCEINLINE char fgBoxVecCompare(const AbsVec& l, const AbsVec& r);
template<> BSS_FORCEINLINE char fgBoxVecCompare<FGBOX_TILEX>(const AbsVec& l, const AbsVec& r) { return SGNCOMPARE(l.x, r.x); }
template<> BSS_FORCEINLINE char fgBoxVecCompare<FGBOX_TILEY>(const AbsVec& l, const AbsVec& r) { return SGNCOMPARE(l.y, r.y); }
template<> BSS_FORCEINLINE char fgBoxVecCompare<FGBOX_TILE>(const AbsVec& l, const AbsVec& r) { char ret = SGNCOMPARE(l.y, r.y); return !ret ? SGNCOMPARE(l.x, r.x) : ret; }
template<> BSS_FORCEINLINE char fgBoxVecCompare<FGBOX_TILE | FGBOX_GROWY>(const AbsVec& l, const AbsVec& r) { char ret = SGNCOMPARE(l.x, r.x); return !ret ? SGNCOMPARE(l.y, r.y) : ret; }

BSS_FORCEINLINE AbsVec fgOrderedBottomRight(const AbsRect& r) { return r.bottomright; }
BSS_FORCEINLINE AbsVec fgOrderedBottomLeft(const AbsRect& r) { return AbsVec{ r.left, r.bottom }; }
BSS_FORCEINLINE AbsVec fgOrderedTopRight(const AbsRect& r) { return AbsVec{ r.right, r.top }; }

template<fgFlag FLAGS, AbsVec(*CORNER)(const AbsRect&)>
char fgOrderedCompare(const AbsVec& l, const fgElement* const& e, const AbsRect* cache)
{
  assert(e->parent != 0);
  AbsRect r;
  ResolveRectCache(e, &r, cache, &e->parent->padding);
  return fgBoxVecCompare<FLAGS&(FGBOX_TILE | FGBOX_GROWY)>(l, CORNER(r));
}

template<fgFlag FLAGS>
fgElement* fgOrderedGet(struct _FG_BOX_ORDERED_ELEMENTS_* self, const AbsRect* area, const AbsRect* cache)
{
  if(!self->ordered.l)
    return 0;
  size_t r = 0;
  if((FLAGS&FGBOX_TILE) != FGBOX_TILE) // Simple case is easy
    r = bss::binsearch_aux_t<const fgElement*, AbsVec, size_t, &bss::CompT_NEQ<char>, -1, const AbsRect*>::template BinarySearchNear<&fgOrderedCompare<FLAGS, fgOrderedBottomRight>>(self->ordered.p, area->topleft, 0, self->ordered.l, cache);
  else
  {
    AbsVec query = (FLAGS&FGBOX_GROWY) ? AbsVec{ area->left, -INFINITY } : AbsVec{ -INFINITY, area->top }; // First we find the target column we want by querying (x,-infinity)
    constexpr auto FN = (FLAGS&FGBOX_GROWY) ? fgOrderedBottomLeft : fgOrderedTopRight;
    r = bss::binsearch_aux_t<const fgElement*, AbsVec, size_t, &bss::CompT_NEQ<char>, -1, const AbsRect*>::template BinarySearchNear<&fgOrderedCompare<FLAGS, FN>>(self->ordered.p, query, 0, self->ordered.l, cache);

    if(r == self->ordered.l) // If this happens, we went past the END of the array
      r = self->ordered.l - 1;
    if(r < self->ordered.l) // Then, if this elements left edge is greater than the area left edge, go back one to get the previous column.
    {
      AbsRect abs;
      ResolveOuterRectCache(self->ordered.p[r], &abs, cache, &self->ordered.p[r]->parent->padding);
      if(((FLAGS&FGBOX_GROWY) && (abs.left > area->left)) || (!(FLAGS&FGBOX_GROWY) && (abs.top > area->top)))
        r -= 1;
    }
    if(r >= self->ordered.l) // Otherwise we went past the START of the array
      r = 0;

    AbsRect abs;
    ResolveOuterRectCache(self->ordered.p[r], &abs, cache, &self->ordered.p[r]->parent->padding);
    query = (FLAGS&FGBOX_GROWY) ? AbsVec{ abs.left, area->top } : AbsVec{ area->left, abs.top }; // Now that we have the correct column, create a query on the y-axis using the left edge of the column.
    r = bss::binsearch_aux_t<const fgElement*, AbsVec, size_t, &bss::CompT_NEQ<char>, -1, const AbsRect*>::template BinarySearchNear<&fgOrderedCompare<FLAGS, FN>>(self->ordered.p, query, 0, self->ordered.l, cache);
  }
  return (r >= self->ordered.l) ? self->ordered.p[0] : self->ordered.p[r];
}
template<fgFlag FLAGS>
BSS_FORCEINLINE fgElement* fgBoxOrder(fgElement* self, const AbsRect* area, const AbsRect* cache) { return fgOrderedGet<FLAGS>(&((fgBox*)self)->order, area, cache); }

template<fgFlag FLAGS>
inline fgElement* fgBoxOrderInject(fgElement* self, const FG_Msg* msg)
{
  AbsRect r = { (FABS)msg->x, (FABS)msg->y, (FABS)msg->x, (FABS)msg->y };
  AbsRect cache;
  ResolveRect(self, &cache);
  return fgOrderedGet<FLAGS>(&((fgBox*)self)->order, &r, &cache);
}

inline fgOrderedDrawGet fgBox_GetDrawOrderFn(fgFlag flags)
{
  switch(flags&(FGBOX_TILE | FGBOX_GROWY))
  {
  case 0:
  case FGBOX_TILEX: return &fgBoxOrder<FGBOX_TILEX>;
  case FGBOX_TILEY: return &fgBoxOrder<FGBOX_TILEY>;
  case FGBOX_TILE: return&fgBoxOrder<FGBOX_TILE>;
  case FGBOX_TILE | FGBOX_GROWY: return &fgBoxOrder<FGBOX_TILE | FGBOX_GROWY>;
  }
  assert(false);
  return 0;
}

fgElement* fgBoxOrderedElement_Get(struct _FG_BOX_ORDERED_ELEMENTS_* self, const AbsRect* target, const AbsRect* area, fgFlag flags)
{
  switch(flags&(FGBOX_TILE | FGBOX_GROWY))
  {
  case FGBOX_TILEX: return fgOrderedGet<FGBOX_TILEX>(self, target, area);
  case FGBOX_TILEY: return fgOrderedGet<FGBOX_TILEY>(self, target, area);
  case FGBOX_TILE: return fgOrderedGet<FGBOX_TILE>(self, target, area);
  case FGBOX_TILE | FGBOX_GROWY: return fgOrderedGet<FGBOX_TILE|FGBOX_GROWY>(self, target, area);
  }
  return 0;
}

void fgBoxRenderDividers(fgElement* self, fgColor color, const AbsRect* area, const AbsRect* drawarea, const fgDrawAuxData* aux, fgElement* begin)
{
  fgElement* next = begin;
  begin = fgLayout_GetPrev(begin);
  if(!begin)
    begin = next;

  AbsRect rbegin;
  ResolveRectCache(begin, &rbegin, area, &self->padding);
  AbsVec center = ResolveVec(&self->transform.center, area);
  AbsVec offset = { 0,0 };
  AbsVec scale = { 1,1 };

  while(next = fgLayout_GetNext(begin)) // We only draw dividers between elements, so once we hit the last one, stop
  {
    AbsRect rnext;
    ResolveRectCache(next, &rnext, area, &self->padding);

    if(self->flags&FGBOX_TILEX)
    {
      float avg = floor((rnext.left + rbegin.right) * 0.5f);
      AbsVec v[2] = { { avg,floor(drawarea->top + self->padding.top)}, { avg,floor(drawarea->bottom - self->padding.bottom)} };
      fgroot_instance->backend.fgDrawLines(v,2, color.color, &offset, &scale, self->transform.rotation, &center, aux);
    }
    else
    {
      float avg = floor((rnext.top + rbegin.bottom) * 0.5f);
      AbsVec v[2] = { { floor(drawarea->left + self->padding.left),avg },{ floor(drawarea->right - self->padding.right),avg } };
      fgroot_instance->backend.fgDrawLines(v, 2, color.color, &offset, &scale, self->transform.rotation, &center, aux);
    }

    begin = next;
    rbegin = rnext;
  }
}

void fgBox_DeselectAll(fgBox* self)
{
  for(size_t i = 0; i < self->selected.l; ++i)
    fgSetFlagStyle(self->selected.p[i], "selected", false);
  ((fgElementArray&)self->selected).Clear();
}

void fgBox_SelectTarget(fgBox* self, fgElement* target)
{
  assert(target != 0 && target->parent == (fgElement*)self);

  fgSetFlagStyle(target, "selected", true);
  ((fgElementArray&)self->selected).Insert(target);
  self->scroll->Selection(target);
}

size_t fgBox_Message(fgBox* self, const FG_Msg* msg)
{
  fgFlag flags = self->scroll.control.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    self->fndraw = 0;
    self->dividercolor.color = 0;
    bss::bssFill(self->selected, 0);
    break;
  case FG_CLONE:
    if(msg->e)
    {
      fgBox* hold = reinterpret_cast<fgBox*>(msg->e);
      memsubcpy<fgBox, fgScrollbar>(hold, self); // We do this first because we have to ensure our messages can process ADDCHILD correctly.
      bss::bssFill(hold->order.ordered);
      bss::bssFill(hold->selected);
      fgScrollbar_Message(&self->scroll, msg);
    }
    return sizeof(fgBox);
  case FG_GETCOLOR:
    if(msg->subtype == FGSETCOLOR_DIVIDER)
      return self->dividercolor.color;
    break;
  case FG_SETCOLOR:
    if(msg->subtype != FGSETCOLOR_DIVIDER)
      break;
    self->dividercolor.color = msg->u;
    return FG_ACCEPT;
  case FG_DRAW:
    if(!self->order.isordered || !self->order.ordered.l)
      fgStandardDraw(*self, (AbsRect*)msg->p, (fgDrawAuxData*)msg->p2, msg->subtype & 1, self->fndraw);
    else
      fgOrderedDraw(*self, (AbsRect*)msg->p, (fgDrawAuxData*)msg->p2, msg->subtype & 1, self->order.ordered.p[self->order.ordered.l - 1]->next, fgBox_GetDrawOrderFn(self->scroll->flags), self->fndraw, self->selected.l > 0 ? self->selected.p[0] : 0);
    return FG_ACCEPT;
  case FG_INJECT:
    if(!self->order.isordered || !self->order.ordered.l)
      return fgStandardInject(*self, (const FG_Msg*)msg->p, (const AbsRect*)msg->p2);
    else
    {
      fgElement* (*fn)(fgElement*, const FG_Msg*);
      switch(self->scroll->flags&(FGBOX_TILE | FGBOX_GROWY))
      {
      case 0:
      case FGBOX_TILEX: fn = &fgBoxOrderInject<FGBOX_TILEX>; break;
      case FGBOX_TILEY: fn = &fgBoxOrderInject<FGBOX_TILEY>; break;
      case FGBOX_TILE: fn = &fgBoxOrderInject<FGBOX_TILE>; break;
      case FGBOX_TILE | FGBOX_GROWY: fn = &fgBoxOrderInject<FGBOX_TILE | FGBOX_GROWY>; break;
      }
      return fgOrderedInject(*self, (const FG_Msg*)msg->p, (const AbsRect*)msg->p2, self->order.ordered.p[self->order.ordered.l - 1]->next, fn, self->selected.l > 0 ? self->selected.p[0] : 0);
    }
  case FG_GETSELECTEDITEM:
    return (msg->u) < self->selected.l ? (size_t)self->selected.p[msg->u] : 0;
  case FG_GETCLASSNAME:
    return (size_t)"Box";
  }

  return fgBoxOrderedElement_Message(&self->order, msg, *self, (fgMessage)&fgScrollbar_Message);
}

void fgBoxCheckOrdered(struct _FG_BOX_ORDERED_ELEMENTS_* self, const FG_Msg* msg, fgElement* element, fgElement* next)
{
  if((msg->e->flags & (FGELEMENT_BACKGROUND | FGELEMENT_NOCLIP)) == FGELEMENT_NOCLIP)
    self->isordered = 0; // If ANY foreground elements are nonclipping, we can't use ordered rendering, because this would require us to maintain a second "nonclipping" sorted array.
  if(self->isordered)
  {
    fgElement* prev = !next ? element->last : next->prev;
    if(msg->e->flags&FGELEMENT_BACKGROUND) // If we're a background element, just make sure we aren't surrounded by foreground elements
    {
      if(next != 0 && !(next->flags&FGELEMENT_BACKGROUND) && prev != 0 && !(prev->flags&FGELEMENT_BACKGROUND))
        self->isordered = 0;
    }
    else if(!((next != 0 && !(next->flags&FGELEMENT_BACKGROUND)) || (prev != 0 && !(prev->flags&FGELEMENT_BACKGROUND)))) // if we are a foreground element and we're touching a foreground element on either side, we're okay. Otherwise, do a probe.
    {
      while(next && (next->flags&FGELEMENT_BACKGROUND)) next = next->next;
      while(prev && (prev->flags&FGELEMENT_BACKGROUND)) prev = prev->prev;
      if((next != 0 && !(next->flags&FGELEMENT_BACKGROUND)) || (prev != 0 && !(prev->flags&FGELEMENT_BACKGROUND)))
        self->isordered = 0; // If we are a foreground element and we just hit another foreground element, there must be at least one background element seperating us, which is invalid.
    }
  }
}

void fgBoxOrderedRemove(struct _FG_BOX_ORDERED_ELEMENTS_* self, fgElement* target)
{
  for(size_t i = 0; i < self->ordered.l; ++i)
    if(self->ordered.p[i] == target)
      ((bss::DynArray<fgElement*>&)self->ordered).Remove(i);
}
void fgBoxOrderedInsert(struct _FG_BOX_ORDERED_ELEMENTS_* self, fgElement* target, fgElement* next)
{
  if(!next || next->flags&FGELEMENT_BACKGROUND)
    ((bss::DynArray<fgElement*>&)self->ordered).Add(target);
  else
  {
    size_t i = self->ordered.l;
    while(i > 0 && self->ordered.p[--i] != next);
    assert(!self->ordered.p || self->ordered.p[i] == next);
    ((bss::DynArray<fgElement*>&)self->ordered).Insert(target, i);
  }
}

size_t fgBoxOrderedElement_Message(struct _FG_BOX_ORDERED_ELEMENTS_* self, const FG_Msg* msg, fgElement* element, fgMessage callback)
{
  fgFlag otherint = (fgFlag)msg->u;
  fgFlag flags = element->flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    bss::bssFill(self->ordered, 0);
    self->isordered = 1;
    self->fixedsize.x = -1;
    self->fixedsize.y = -1;
    break;
  case FG_SETFLAG: // Do the same thing fgElement does to resolve a SETFLAG into SETFLAGS
    otherint = bss::bssSetBit<fgFlag>(flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((otherint^flags) & FGBOX_LAYOUTMASK)
    { // handle a layout flag change
      size_t r = callback(element, msg); // we have to actually set the flags first before resetting the layout
      fgSubMessage(element, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTRESET, 0, 0);
      return r;
    }
    break;
  case FG_LAYOUTFUNCTION:
    if(element->flags&FGBOX_DISTRIBUTE) // DISTRIBUTE overrides the tiling flags
      return fgDistributeLayout(element, (const FG_Msg*)msg->p, element->flags&FGBOX_LAYOUTMASK, (AbsVec*)msg->p2);
    if(element->flags&(FGBOX_TILEX | FGBOX_TILEY))
      return fgTileLayout(element, (const FG_Msg*)msg->p, element->flags&FGBOX_LAYOUTMASK, (AbsVec*)msg->p2);
    break; // If no layout flags are specified, fall back to default layout behavior.
  case FG_REMOVECHILD:
  {
    fgBoxOrderedRemove(self, msg->e);
    size_t r = callback(element, msg);
    self->isordered = checkIsOrdered(element->root);
    return r;
  }
  case FG_REORDERCHILD:
    assert(msg->p != 0);
    if(callback(element, msg) == FG_ACCEPT)
    {
      fgElement* next = msg->e->next;
      fgBoxCheckOrdered(self, msg, element, next);

      if(!(msg->e->flags&FGELEMENT_BACKGROUND))
      {
        if(self->isordered) // If we are still ordered, we have to remove this from the array and add it back in at the proper location
        {
          fgBoxOrderedRemove(self, msg->e);
          fgBoxOrderedInsert(self, msg->e, next);
        }
        else
          self->ordered.l = 0;
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_ADDCHILD:
    assert(msg->p != 0);
    if(callback(element, msg) == FG_ACCEPT)
    {
      fgElement* next = msg->e->next;
      fgBoxCheckOrdered(self, msg, element, next);

      if(!(msg->e->flags&FGELEMENT_BACKGROUND))
      {
        if(self->isordered) // If we are still ordered, we have to remove this from the array and add it back in at the proper location
          fgBoxOrderedInsert(self, msg->e, next);
        else
          self->ordered.l = 0;
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_ADDITEM:
    if(!self->isordered || msg->subtype != FGITEM_ELEMENT)
      return 0; // Can't set anything if we aren't ordered
    else
    {
      fgElement* next = (msg->u2 < self->ordered.l) ? self->ordered.p[msg->u2] : 0;
      element->AddChild(msg->e, next);
      return FG_ACCEPT;
    }
  case FG_REMOVEITEM:
    if(!self->isordered)
      return 0; // Can't remove by index if we aren't ordered
    if(msg->u < self->ordered.l)
    {
      VirtualFreeChild(self->ordered.p[msg->u]);
      return FG_ACCEPT;
    }
    return 0;
  case FG_GETITEM:
    if(!self->isordered)
      return 0; // If this isn't ordered we can't get an item by index
    if(msg->subtype == FGITEM_COUNT)
      return self->ordered.l;
    if(msg->subtype == FGITEM_LOCATION) // This means to query for the nearest element to the given coordinates.
    {
      AbsRect r = { (FABS)msg->x, (FABS)msg->y, (FABS)msg->x, (FABS)msg->y };
      AbsRect cache;
      ResolveRect(element, &cache);

      switch(element->flags&(FGBOX_TILE | FGBOX_GROWY))
      {
      case 0:
      case FGBOX_TILEX: return (size_t)fgOrderedGet<FGBOX_TILEX>(self, &r, &cache);
      case FGBOX_TILEY: return (size_t)fgOrderedGet<FGBOX_TILEY>(self, &r, &cache);
      case FGBOX_TILE: return (size_t)fgOrderedGet<FGBOX_TILE>(self, &r, &cache);
      case FGBOX_TILE | FGBOX_GROWY: return (size_t)fgOrderedGet<FGBOX_TILE | FGBOX_GROWY>(self, &r, &cache);
      }
    }
    else if(size_t(msg->i) < self->ordered.l)
      return (size_t)self->ordered.p[msg->i];
    return 0;
  case FG_SETITEM:
    if(!self->isordered || msg->subtype != FGITEM_ELEMENT)
      return 0; // Can't set anything if we aren't ordered
    if(msg->u2 < self->ordered.l)
    {
      fgElement* next = self->ordered.p[msg->u2]->next;
      VirtualFreeChild(self->ordered.p[msg->u2]);
      element->AddChild(msg->e, next);
      return FG_ACCEPT;
    }
    return 0;
  }

  return callback(element, msg);
}