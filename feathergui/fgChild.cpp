// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgChild.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include "feathercpp.h"
#include <math.h>
#include <limits.h>

typedef bss_util::cDynArray<fgChild*, FG_UINT> fgSkinRefArray;

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
  fgChild_InternalSetup(self, flags, parent, element, (FN_DESTROY)&fgChild_Destroy, (FN_MESSAGE)&fgChild_Message);
}

void FG_FASTCALL fgChild_Destroy(fgChild* self)
{
  assert(self != 0);
  if(fgFocusedWindow == self)
    fgChild_VoidMessage(self, FG_LOSTFOCUS, 0);

  fgChild_Clear(self);
  fgChild_SetParent(self->root, 0);
  ((fgSkinRefArray&)self->skinrefs).~cDynArray();
}

void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent)
{
  assert(self != 0);
  if(self->parent == parent)
    return;

  if(self->parent != 0)
  {
    fgChild_SubMessage(self->parent, FG_LAYOUTCHANGE, FGCHILD_LAYOUTREMOVE, self, 0);
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
    fgChild_SubMessage(parent, FG_LAYOUTCHANGE, FGCHILD_LAYOUTADD, self, 0);
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
  fgChild_VoidAuxMessage(child, FG_SETSTYLE, &layout->style.style, 2);

  for(FG_UINT i = 0; i < layout->children.l; ++i)
    fgChild_LoadLayout(child, DynGetP<fgClassLayoutArray>(layout->children, i), mapping);

  return child;
}

typedef fgChild* (*FN_MAPPING)(const char*, fgFlag, fgChild*, fgElement*);

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
      fgChild_SubMessage(self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTMOVE, msg->other, msg->otheraux);
    else if(msg->otheraux) // This was either internal or propagated down, in which case we must keep propagating it down so long as something changed.
    {
      fgChild* ref = !msg->other ? self : (fgChild*)msg->other;
      fgChild* cur = self->root;
      char diff;
      while(hold = cur)
      {
        cur = cur->next;
        diff = fgChild_PotentialResize(hold);
        
        fgChild_VoidAuxMessage(hold, FG_MOVE, ref, diff & msg->otheraux);
      }

      if(msg->otheraux & 0b10000110)
        fgChild_SubMessage(self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTRESIZE, (void*)msg->otheraux, 0);
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
      fgChild_SubMessage(self->parent, FG_LAYOUTCHANGE, (self->flags & FGCHILD_BACKGROUND) ? FGCHILD_LAYOUTADD : FGCHILD_LAYOUTREMOVE, self, 0);
    if((change&FGCHILD_EXPAND)&self->flags) // If we change the expansion flags, we must recalculate every single child in our layout provided one of the expansion flags is actually set
      fgChild_SubMessage(self, FG_LAYOUTCHANGE, FGCHILD_LAYOUTRESET, 0, 0);
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
        fgChild* old = !self->prev ? self->next : self->prev; // self->next is always valid if we switched places and self->prev is 0.
        if(!LList_ChangeOrderAll(self)) // Only send a notification if we actually swapped places.
        {
          assert(old != 0);
          FG_Msg m;
          m.type = FG_SETORDER;
          m.other1 = self;
          m.other2 = old;
          (*fgroot_instance->behaviorhook)(self->parent, &m);
        }
      }
    }
    else
    {
      FG_Msg m = *msg;
      m.type = FG_LAYOUTCHANGE;
      m.subtype = FGCHILD_LAYOUTREORDER;
      (*fgroot_instance->behaviorhook)(self, &m);
    }
    return 0; // Otherwise it's just a notification
  case FG_SETPARENT:
    fgChild_SetParent(self, (fgChild*)msg->other);
    fgChild_VoidMessage(self, FG_SETSKIN, 0); // re-evaluate our skin
    if(msg->other)
      fgChild_VoidAuxMessage((fgChild*)msg->other, FG_MOVE, 0, fgChild_PotentialResize(self));
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
    return fgChild_VoidMessage((fgChild*)msg->other, FG_SETPARENT, 0);
  case FG_LAYOUTCHANGE:
  {
    CRect area = self->element.area;
    FG_Msg m = { 0 };
    m.type = FG_LAYOUTFUNCTION;
    m.other1 = (void*)msg;
    m.other2 = &area;
    if(fgChild_PassMessage(self, &m) != 0)
      fgChild_VoidMessage(self, FG_SETAREA, &area);
    return 0;
  }
  case FG_LAYOUTFUNCTION:
    return fgLayout_Default(self, (const FG_Msg*)msg->other1, (CRect*)msg->other2);
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = (fgLayout*)msg->other1;
    if(!layout)
      return 1;

    FN_MAPPING mapping = (FN_MAPPING)msg->other2;
    if(!mapping) mapping = &fgLayoutLoadMapping;

    for(FG_UINT i = 0; i < layout->layout.l; ++i)
      fgChild_LoadLayout(self, DynGetP<fgClassLayoutArray>(layout->layout, i), mapping);
  }
    return 0;
  case FG_CLONE:
  {
    hold = (fgChild*)msg->other;
    if(!hold)
      hold = (fgChild*)malloc(sizeof(fgChild));
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
      
      int index; // First we check if we're a skin reference, and if we are, check if there is a skin for us.
      for(index = 0; index < self->parent->skinrefs.l; ++index)
        if(DynGet<fgSkinRefArray>(self->parent->skinrefs, index) == self)
          break;
      if(index < self->parent->skinrefs.l && self->parent->skin != 0)
      {
        index -= self->parent->prechild;
        for(FG_UINT i = 0; i < self->parent->skin->subskins.l; ++i)
        {
          fgSkin* skin = DynGetP<fgSubskinArray>(self->parent->skin->subskins, index);
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
      return (*fgroot_instance->behaviorhook)(self->parent, msg);
    return 0;
  case FG_SETSKIN:
  {
    fgSkin* skin = (fgSkin*)(!msg->other ? (void*)fgChild_VoidMessage(self, FG_GETSKIN, 0) : msg->other);
    if(self->skin == skin) break; // no change in skin
    if(self->skin != 0)
    {
      for(FG_UINT i = self->prechild; i < self->skinrefs.l; ++i)
        fgChild_VoidMessage(self, FG_REMOVECHILD, DynGet<fgSkinRefArray>(self->skinrefs, i));
    }
    self->skinrefs.l = self->prechild;
    self->skin = skin;
    if(self->skin != 0)
    {
      FN_MAPPING mapping = (FN_MAPPING)msg->other2;
      if(!mapping) mapping = &fgLayoutLoadMapping;

      for(FG_UINT i = 0; i < self->skin->children.l; ++i)
      {
        fgStyleLayout* layout = DynGetP<fgStyleLayoutArray>(self->skin->children, i);
        fgChild* child = (*mapping)(layout->name, layout->flags, self, &layout->element);
        fgChild_VoidAuxMessage(child, FG_SETSTYLE, &layout->style, 2);
        ((fgSkinRefArray&)self->skinrefs).Add(child);
      }

      fgChild* cur = self->root;
      while(cur)
      {
        fgChild_VoidMessage(cur, FG_SETSKIN, 0); // This will automatically set any subskins we have if necessary.
        cur = cur->next;
      }
      fgChild_VoidAuxMessage(self, FG_SETSTYLE, (void*)&self->skin->style, 2);
    }
    fgChild_IntMessage(self, FG_SETSTYLE, -1, 1);
  }
    return 0;
  case FG_SETSTYLE:
  {
    fgStyle* style = 0;
    if(msg->otheraux != 2)
    {
      FG_UINT index = (!msg->otheraux ? fgStyle_GetName((const char*)msg->other) : msg->otherint);

      if(index == -1)
        index = fgChild_VoidMessage(self, FG_GETSTYLE, 0);
      else
        self->style = index;

      if(self->skin != 0 && index < self->skin->styles.l)
        style = DynGetP<fgStyleArray>(self->skin->styles, index);

      fgChild* cur = self->root;
      while(cur)
      {
        fgChild_IntMessage(cur, FG_SETSTYLE, -1, 1); // Forces the child to recalculate the style inheritance
        cur = cur->next;
      }
    }
    else
      style = (fgStyle*)msg->other;

    if(!style)
      return 1;

    fgStyleMsg* cur = style->styles;
    while(cur)
    {
      (*fgroot_instance->behaviorhook)(self, &cur->msg);
      cur = cur->next;
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
    fgStandardDraw(self, (AbsRect*)msg->other, INT_MAX);
    return 0;
  case FG_GETNAME:
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

BSS_FORCEINLINE fgChild*& fgChild_prev(fgChild* p) { return p->prev; }
BSS_FORCEINLINE fgChild*& fgChild_prevclip(fgChild* p) { return p->prevclip; }

BSS_FORCEINLINE fgChild*& fgChild_next(fgChild* p) { return p->next; }
BSS_FORCEINLINE fgChild*& fgChild_nextclip(fgChild* p) { return p->nextclip; }

template<fgChild*&(*PREV)(fgChild*), fgChild*&(*NEXT)(fgChild*)>
void FG_FASTCALL LList_Remove(fgChild* self, fgChild** root, fgChild** last)
{
  assert(self != 0);
  if(PREV(self) != 0) NEXT(PREV(self)) = NEXT(self);
  else *root = NEXT(self);
  if(NEXT(self) != 0) PREV(NEXT(self)) = PREV(self);
  else *last = PREV(self);
}

void FG_FASTCALL LList_RemoveAll(fgChild* self)
{
  LList_Remove<fgChild_prev, fgChild_next>(self, &self->parent->root, &self->parent->last); // Remove ourselves from our parent
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_Remove<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Remove<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}

template<fgChild*&(*PREV)(fgChild*), fgChild*&(*NEXT)(fgChild*)>
void FG_FASTCALL LList_Insert(fgChild* self, fgChild* cur, fgChild* prev, fgChild** root, fgChild** last)
{
  NEXT(self) = cur;
  PREV(self) = prev;
  if(prev) NEXT(prev) = self;
  else *root = self; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) PREV(cur) = self;
  else *last = self; // Cur is null if we are at the end of the list, so update last
}

template<fgChild*&(*PREV)(fgChild*), fgChild*&(*NEXT)(fgChild*)>
void FG_FASTCALL LList_Add(fgChild* self, fgChild** root, fgChild** last)
{
  fgChild* cur = *root;
  fgChild* prev = 0; // Sadly the elegant pointer to pointer method doesn't work for doubly linked lists.
  assert(self != 0 && root != 0);
  if(!cur) // We do this check up here because we'd have to do it for the end append check below anyway so we might as well utilize it!
    LList_Insert<PREV, NEXT>(self, 0, 0, root, last);
  else if(self->order < (*last)->order) // shortcut for appending to the end of the list
    LList_Insert<PREV, NEXT>(self, 0, *last, root, last);
  else
  {
    while(cur != 0 && (self->order < cur->order))
    {
      prev = cur;
      cur = NEXT(cur);
    }
    LList_Insert<PREV, NEXT>(self, cur, prev, root, last);
  }
}

void FG_FASTCALL LList_AddAll(fgChild* self)
{
  LList_Add<fgChild_prev, fgChild_next>(self, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_Add<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Add<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}

template<fgChild*&(*PREV)(fgChild*), fgChild*&(*NEXT)(fgChild*)>
char FG_FASTCALL LList_ChangeOrder(fgChild* self, fgChild** root, fgChild** last)
{
  fgChild* cur = NEXT(self);
  fgChild* prev = PREV(self);
  while(cur != 0 && (self->order < cur->order))
  {
    prev = cur;
    cur = NEXT(cur);
  }
  while(prev != 0 && (self->order > prev->order))
  {
    cur = prev;
    prev = PREV(prev);
  }

  if(cur == NEXT(self)) { assert(prev == PREV(self)); return 1; } // we didn't move anywhere
  LList_Remove<PREV, NEXT>(self, root, last);
  LList_Insert<PREV, NEXT>(self, cur, prev, root, last);
  return 0;
}

char FG_FASTCALL LList_ChangeOrderAll(fgChild* self)
{
  char r = LList_ChangeOrder<fgChild_prev, fgChild_next>(self, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_ChangeOrder<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_ChangeOrder<fgChild_prevclip, fgChild_nextclip>(self, &self->parent->rootclip, &self->parent->lastclip);
  }
  return r;
}

size_t FG_FASTCALL fgChild_VoidMessage(fgChild* self, unsigned char type, void* data)
{
  FG_Msg aux = { 0 };
  aux.type = type;
  aux.other = data;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &aux);
}

size_t FG_FASTCALL fgChild_VoidAuxMessage(fgChild* self, unsigned char type, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

size_t FG_FASTCALL fgChild_IntMessage(fgChild* self, unsigned char type, ptrdiff_t data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.otherint = data;
  msg.otherintaux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

FG_EXTERN size_t FG_FASTCALL fgChild_PassMessage(fgChild* self, const FG_Msg* msg)
{
  return (*fgroot_instance->behaviorhook)(self, msg);
}

FG_EXTERN size_t FG_FASTCALL fgChild_SubMessage(fgChild* self, unsigned char type, unsigned char subtype, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = subtype;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

size_t FG_FASTCALL fgLayout_Default(fgChild* self, const FG_Msg* msg, CRect* area)
{
  if(!(self->flags & FGCHILD_EXPAND))
    return 0;

  fgChild* child = (fgChild*)msg->other;

  switch(msg->subtype)
  {
  case FGCHILD_LAYOUTREORDER:
  case FGCHILD_LAYOUTRESIZE:
    break;
  case FGCHILD_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGCHILD_LAYOUTREMOVE;
    size_t dim = fgLayout_Default(self, &m, area);
    m.subtype = FGCHILD_LAYOUTADD;
    dim |= fgLayout_Default(self, &m, area);
    return dim;
  }
  case FGCHILD_LAYOUTADD:
  {
    if(child->flags & FGCHILD_BACKGROUND)
      break;
    char dim = 0;
    if(self->flags & FGCHILD_EXPANDX)
      dim |= fgLayout_ExpandX(area, child);
    if(self->flags & FGCHILD_EXPANDY)
      dim |= fgLayout_ExpandY(area, child);
    return dim;
  }
  case FGCHILD_LAYOUTREMOVE:
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
    {
      FG_Msg m = *msg;
      m.subtype = FGCHILD_LAYOUTRESET;
      m.other = child;
      return fgLayout_Default(self, &m, area);
    }
  }
  case FGCHILD_LAYOUTRESET:
  {
    fgChild* hold = self->root;
    if(self->flags & FGCHILD_EXPANDX)
      area->right.abs = area->left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
    if(self->flags & FGCHILD_EXPANDY)
      area->bottom.abs = area->top.abs;
    char dim = 0;

    while(hold)
    {
      if(hold != child)
      {
        if(self->flags & FGCHILD_EXPANDX)
          dim |= fgLayout_ExpandX(area, hold);
        if(self->flags & FGCHILD_EXPANDY)
          dim |= fgLayout_ExpandY(area, hold);
      }
      hold = hold->next;
    }

    return dim;
  }
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

  switch(msg->subtype)
  {
  case FGCHILD_LAYOUTMOVE:
    break;
  case FGCHILD_LAYOUTRESIZE:
    if((((msg->otherint&0b10) && !(self->flags&FGCHILD_EXPANDX)) || ((msg->otherint & 0b100) && !(self->flags&FGCHILD_EXPANDY))) && (axes & 3))
      fgLayout_TileReorder(0, self->root, axis, max, 0.0f);
    break;
  case FGCHILD_LAYOUTREORDER:
  {
    fgChild* old = (fgChild*)msg->other2;
    child = child->order < old->order ? child : old; // Get lowest child
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
  }
    break;
  case FGCHILD_LAYOUTADD:
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  case FGCHILD_LAYOUTREMOVE:
    fgLayout_TileReorder(child->prev, child->next, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Distribute(fgChild* self, const FG_Msg* msg, char axis)
{
  /*switch(msg->type)
  {
  case FGCHILD_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGCHILD_LAYOUTRESIZE:
    break;
  case FGCHILD_LAYOUTREORDER:
    //child = child->order<msg->other->order?child:msg->other; // Get lowest child
    fgLayout_DistributeReorder(child->prev, child, axis, num);
    break;
  case FGCHILD_LAYOUTADD:
    fgLayout_DistributeReorder(child->prev, child, axis, num);
    break;
  case FGCHILD_LAYOUTREMOVE:
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
  ((fgSkinRefArray&)self->skinrefs).Insert(child, self->prechild++);
}