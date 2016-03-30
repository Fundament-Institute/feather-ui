// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgChild.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include "feathercpp.h"
#include <math.h>
#include <limits.h>

template<typename U, typename V>
BSS_FORCEINLINE char CompPairInOrder(const std::pair<U, V>& l, const std::pair<U, V>& r) { char ret = SGNCOMPARE(l.first, r.first); return !ret ? SGNCOMPARE(l.second, r.second) : ret; }

typedef bss_util::cDynArray<fgChild*, FG_UINT> fgSkinRefArray;
bss_util::cAVLtree<std::pair<fgChild*, unsigned short>, void, &CompPairInOrder> fgListenerList;

void FG_FASTCALL fgChild_InternalSetup(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*))
{
  assert(self != 0);
  memset(self, 0, sizeof(fgChild));
  self->destroy = destroy;
  self->free = 0;
  self->message = message;
  self->flags = flags;
  self->style = (FG_UINT)-1;
  if(element) self->element = *element;
  fgSendMsg<FG_CONSTRUCT>(self);
  fgSendMsg<FG_SETPARENT, void*, void*>(self, parent, prev);
}

void FG_FASTCALL fgChild_Init(fgChild* BSS_RESTRICT self, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(self, flags, parent, prev, element, (FN_DESTROY)&fgChild_Destroy, (FN_MESSAGE)&fgChild_Message);
}

void FG_FASTCALL fgChild_Destroy(fgChild* self)
{
  assert(self != 0);
  if(fgFocusedWindow == self)
    fgSendMsg<FG_LOSTFOCUS>(self);

  fgChild_Clear(self);
  fgChild_SetParent(self, 0, 0);
  reinterpret_cast<fgSkinRefArray&>(self->skinrefs).~cDynArray();
  fgChild_ClearListeners(self);
}

void FG_FASTCALL fgChild_SetParent(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev)
{
  assert(self != 0);
  if(self->parent == parent)
  {
    if(self->prev != prev)
    {
      assert(self->parent != 0);
      assert(prev->parent == self->parent);
      fgChild* old = self->prev;
      LList_RemoveAll(self);
      LList_InsertAll(self, prev);
      fgChild_SubMessage(self->parent, FG_LAYOUTCHANGE, FGCHILD_LAYOUTREORDER, self, (ptrdiff_t)old);
    }
    return;
  }
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
    LList_InsertAll(self, prev);
    fgChild_SubMessage(parent, FG_LAYOUTCHANGE, FGCHILD_LAYOUTADD, self, 0);
  }
}

char FG_FASTCALL fgChild_PotentialResize(fgChild* self)
{
  return ((self->element.area.left.rel != 0 || self->element.area.right.rel != 0) << 3) // If you have nonzero relative coordinates, a resize will cause a move
    | ((self->element.area.top.rel != 0 || self->element.area.bottom.rel != 0) << 4)
    | ((self->element.area.left.rel != self->element.area.right.rel) << 1) // If you have DIFFERENT relative coordinates, a resize will cause a move AND a resize for you.
    | ((self->element.area.top.rel != self->element.area.bottom.rel) << 2);
}

fgChild* FG_FASTCALL fgChild_LoadLayout(fgChild* parent, fgChild* prev, fgClassLayout* layout, FN_MAPPING mapping)
{
  fgChild* child = (*mapping)(layout->style.name, layout->style.flags, parent, prev, &layout->style.element);
  fgChild_VoidMessage(child, FG_SETSTYLE, &layout->style.style, 2);

  fgChild* p = 0;
  for(FG_UINT i = 0; i < layout->children.l; ++i)
    p = fgChild_LoadLayout(child, p, layout->children.p + i, mapping);

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
      fgChild_SubMessage(self->parent, FG_MOVE, msg->subtype, self, msg->otheraux | 1);
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
        
        fgChild_SubMessage(hold, FG_MOVE, msg->subtype, ref, diff & msg->otheraux);
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
      if(diff)
      {
        fgChild_MouseMoveCheck(self);
        memcpy(&self->element.area, area, sizeof(CRect));
        fgChild_MouseMoveCheck(self);

        fgChild_SubMessage(self, FG_MOVE, FG_SETAREA, 0, diff);
      }
    }
    return 0;
  case FG_SETELEMENT:
    if(!msg->other)
      return 1;
    {
      fgElement* element = (fgElement*)msg->other;
      char diff = CompareElements(&self->element, element);
      if(diff)
      {
        fgChild_MouseMoveCheck(self);
        memcpy(&self->element, element, sizeof(fgElement));
        fgChild_MouseMoveCheck(self);

        fgChild_SubMessage(self, FG_MOVE, FG_SETELEMENT, 0, diff);
      }
    }
    return 0;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
  {
    fgFlag oldflags = self->flags;
    fgFlag change = self->flags ^ (fgFlag)otherint;
    self->flags = (fgFlag)otherint;
    if(change&FGCHILD_IGNORE || change&FGCHILD_NOCLIP)
    {
      self->flags = oldflags;
      fgChild* old = self->prev;
      LList_RemoveAll(self);
      self->flags = (fgFlag)otherint;
      LList_InsertAll(self, old);
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

      if(diff)
      {
        fgChild_MouseMoveCheck(self);
        memcpy(&self->margin, margin, sizeof(AbsRect));
        fgChild_MouseMoveCheck(self);

        fgChild_SubMessage(self, FG_MOVE, FG_SETMARGIN, 0, diff);
      }
    }
    return 0;
  case FG_SETPADDING:
    if(!msg->other)
      return 1;
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareAbsRects(&self->padding, padding);

      if(diff)
      {
        fgChild_MouseMoveCheck(self);
        memcpy(&self->padding, padding, sizeof(AbsRect));
        fgChild_MouseMoveCheck(self);

        fgChild_SubMessage(self, FG_MOVE, FG_SETPADDING, 0, diff);
      }
    }
    return 0;
  case FG_SETPARENT:
    fgChild_SetParent(self, (fgChild*)msg->other, (fgChild*)msg->other2);
    fgSendMsg<FG_SETSKIN>(self); // re-evaluate our skin
    fgChild_SubMessage(self, FG_MOVE, FG_SETPARENT, 0, fgChild_PotentialResize(self));
    return 0;
  case FG_ADDCHILD:
    hold = (fgChild*)msg->other;
    if(!hold)
      return 1;
    return fgSendMsg<FG_SETPARENT, void*>(hold, self);
  case FG_REMOVECHILD:
    hold = (fgChild*)msg->other;
    if(!msg->other || hold->parent != self)
      return 1;
    return fgSendMsg<FG_SETPARENT>((fgChild*)msg->other);
  case FG_LAYOUTCHANGE:
  {
    CRect area = self->element.area;
    FG_Msg m = { 0 };
    m.type = FG_LAYOUTFUNCTION;
    m.other = (void*)msg;
    m.other2 = &area;
    if(fgChild_PassMessage(self, &m) != 0)
    {
      if(self->flags&FGCHILD_EXPANDX)
        area.right.abs += self->padding.left + self->padding.right;
      if(self->flags&FGCHILD_EXPANDY)
        area.bottom.abs += self->padding.top + self->padding.bottom;
      fgSendMsg<FG_SETAREA, void*>(self, &area);
    }
    return 0;
  }
  case FG_LAYOUTFUNCTION:
    return fgLayout_Default(self, (const FG_Msg*)msg->other, (CRect*)msg->other2);
  case FG_LAYOUTINDEX:
    hold = (fgChild*)msg->other;
    return !hold->prev ? 0 : (hold->prev->index + 1);
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = (fgLayout*)msg->other;
    if(!layout)
      return 1;

    FN_MAPPING mapping = (FN_MAPPING)msg->other2;
    if(!mapping) mapping = &fgLayoutLoadMapping;

    fgChild* prev = self->last;
    for(FG_UINT i = 0; i < layout->layout.l; ++i)
      prev = fgChild_LoadLayout(self, prev, layout->layout.p + i, mapping);
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
      fgSendMsg<FG_ADDCHILD, void*>(hold, (fgChild*)fgSendMsg<FG_CLONE>(cur));
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
        if(self->parent->skinrefs.p[index] == self)
          break;
      if(index < self->parent->skinrefs.l && self->parent->skin != 0)
      {
        index -= self->parent->prechild;
        if(index < 0) index = -(index + self->parent->prechild + 1); // reverse negative indices so that adding a prechild to a widget does not change existing indices
        for(FG_UINT i = 0; i < self->parent->skin->subskins.l; ++i)
        {
          fgSkin* skin = self->parent->skin->subskins.p + i;
          if(skin->index == index)
            return (size_t)skin;
        }
      }

      // If not we start propagating up
      return fgSendMsg<FG_GETSKIN, void*>(self->parent, self);
    }
    else if(self->skin != 0)
    {
      hold = (fgChild*)msg->other;
      const char* name = hold->GetName();
      if(name)
      {
        fgSkin* skin = fgSkin_GetSkin(self->skin, name);
        if(skin != 0)
          return (size_t)skin;
      }
      name = hold->GetClassName();
      
      fgSkin* skin = fgSkin_GetSkin(self->skin, name);
      if(skin != 0)
        return (size_t)skin;
    }
    if(self->parent != 0)
      return (*fgroot_instance->behaviorhook)(self->parent, msg);
    return 0;
  case FG_SETSKIN:
  {
    fgSkin* skin = (fgSkin*)(!msg->other ? (void*)fgSendMsg<FG_GETSKIN>(self) : msg->other);
    if(self->skin != skin) // only bother changing the skin if there's stuff to change
    {
      if(self->skin != 0)
      {
        for(FG_UINT i = self->prechild; i < self->skinrefs.l; ++i)
          fgSendMsg<FG_REMOVECHILD, void*>(self, self->skinrefs.p[i]);
      }
      self->skinrefs.l = self->prechild;
      self->skin = skin;
      if(self->skin != 0)
      {
        FN_MAPPING mapping = (FN_MAPPING)msg->other2;
        if(!mapping) mapping = &fgLayoutLoadMapping;

        fgChild* child = self->last;
        for(FG_UINT i = 0; i < self->skin->children.l; ++i)
        {
          fgStyleLayout* layout = self->skin->children.p + i;
          child = (*mapping)(layout->name, layout->flags, self, child, &layout->element);
          fgChild_VoidMessage(child, FG_SETSTYLE, &layout->style, 2);
          ((fgSkinRefArray&)self->skinrefs).Add(child);
        }
        fgChild_VoidMessage(self, FG_SETSTYLE, (void*)&self->skin->style, 2);
      }
    }

    fgChild* cur = self->root; // However, if we get a SETSKIN message, we must pass it to ALL our children no matter what, because they may have gotten new skins we don't know about.
    while(cur)
    {
      fgSendMsg<FG_SETSKIN, void*, void*>(cur, 0, msg->other2); // This will automatically set any subskins we have if necessary.
      cur = cur->next;
    }
    fgChild_IntMessage(self, FG_SETSTYLE, -1, 1);
  }
    return 0;
  case FG_SETSTYLE:
  {
    fgStyle* style = 0;
    if(msg->otheraux != 2)
    {
      size_t index = (!msg->otheraux ? fgStyle_GetName((const char*)msg->other) : (size_t)msg->otherint);

      if(index == -1)
        index = fgSendMsg<FG_GETSTYLE>(self);
      else
        self->style = (FG_UINT)index;

      if(self->skin != 0 && index < self->skin->styles.l)
        style = self->skin->styles.p + index;

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
    return (self->style == (FG_UINT)-1 && self->parent != 0) ? fgSendMsg<FG_GETSTYLE>(self->parent) : self->style;
  case FG_GOTFOCUS:
    if(self->lastfocus) {
      fgChild* hold = self->lastfocus;
      self->lastfocus = 0;
      if(!fgSendMsg<FG_GOTFOCUS>(hold))
        return 0;
    }
    if(self->parent)
      return fgSendMsg<FG_GOTFOCUS>(self->parent);
    break;
  case FG_DRAG:
  {
    fgroot_instance->drag = (fgChild*)msg->other;
    FG_Msg m = *(FG_Msg*)msg->other2; // Immediately trigger an FG_DRAGGING message to set the cursor
    m.type = FG_DRAGGING;
    fgChild_PassMessage(self, &m);
    return 0;
  }
  case FG_DRAGGING:
    fgSetCursor(FGCURSOR_NO, 0);
    return 1;
  case FG_DROP:
    fgSetCursor(FGCURSOR_ARROW, 0);
    return 1;
  case FG_DRAW:
    fgStandardDraw(self, (AbsRect*)msg->other, msg->otheraux, INT_MAX);
    return 0;
  case FG_GETNAME:
    return 0;
  case FG_GETDPI:
    return self->parent ? fgSendMsg<FG_GETDPI>(self->parent) : 0;
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
  assert(out != 0);
  if(!self->parent)
  {
    const CRect& v = self->element.area;
    out->left = v.left.abs;
    out->top = v.top.abs;
    out->right = v.right.abs;
    out->bottom = v.bottom.abs;
    return;
  }

  AbsRect last;
  ResolveRect(self->parent, &last);
  ResolveRectCache(self, out, &last, (self->flags & FGCHILD_BACKGROUND) ? 0 : &self->padding);
}

void FG_FASTCALL ResolveRectCache(const fgChild* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding)
{
  AbsRect replace;
  //if(!(self->flags & FGCHILD_BACKGROUND))
  if(padding != 0)
  {
    replace.left = last->left + padding->left;
    replace.top = last->top + padding->top;
    replace.right = last->right - padding->right;
    replace.bottom = last->bottom - padding->bottom;
    //last = &replace;
  }

  AbsVec center = { self->element.center.x.abs, self->element.center.y.abs };
  const CRect* v = &self->element.area;
  assert(out != 0 && self != 0 && last != 0);
  out->left = lerp(last->left, last->right, v->left.rel) + v->left.abs;
  out->top = lerp(last->top, last->bottom, v->top.rel) + v->top.abs;
  out->right = lerp(last->left, last->right, v->right.rel) + v->right.abs;
  out->bottom = lerp(last->top, last->bottom, v->bottom.rel) + v->bottom.abs;

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
    fgChild_MouseMoveCheck(self); // we have to check if the mouse intersects the child BEFORE we remove it so we can properly resolve relative coordinates.
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

void FG_FASTCALL LList_InsertAll(fgChild* BSS_RESTRICT self, fgChild* BSS_RESTRICT prev)
{
  fgChild* next = !prev ? self->parent->root : prev->next;
  LList_Insert<fgChild_prev, fgChild_next>(self, next, prev, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGCHILD_IGNORE))
  {
    if(self->flags&FGCHILD_NOCLIP)
      LList_Insert<fgChild_prevclip, fgChild_nextclip>(self, next, prev, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Insert<fgChild_prevclip, fgChild_nextclip>(self, next, prev, &self->parent->rootclip, &self->parent->lastclip);
    fgChild_MouseMoveCheck(self); // This has to be AFTER so we can properly resolve relative coordinates
  }
}

size_t FG_FASTCALL fgChild_VoidMessage(fgChild* self, unsigned char type, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

size_t FG_FASTCALL fgChild_IntMessage(fgChild* self, unsigned char type, ptrdiff_t data, size_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.otherint = data;
  msg.otheraux = aux;
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

    while(hold) // O(n) by necessity
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

  while(cur) // O(n) complexity, but should break after re-ordering becomes unnecessary.
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
    child = child->index < old->index ? child : old; // Get lowest child
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
  }
    break;
  case FGCHILD_LAYOUTADD:
    child->index = (int)fgSendMsg<FG_LAYOUTINDEX, void*>(self, child);
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

fgChild* FG_FASTCALL fgChild_GetChildUnderMouse(fgChild* self, int x, int y, AbsRect* cache)
{
  ResolveRect(self, cache);
  AbsRect child;
  fgChild* cur = self->root; // we have to go through the whole child list because you can have FG_IGNORE on but still be a hover target, espiecally if you're an fgText control inside a list.
  while(cur != 0)
  {
    if(!(cur->flags&FGCHILD_BACKGROUND))
    {
      ResolveRectCache(cur, &child, cache, &cur->padding); // this is only done for non background elements, so we always pass in the padding.
      if(HitAbsRect(&child, (FABS)x, (FABS)y))
        return cur;
    }
    cur = cur->next;
  }
  return 0;
}

void FG_FASTCALL fgChild_MouseMoveCheck(fgChild* self)
{
  if(!self->parent || (self->flags&FGCHILD_IGNORE))
    return; // If you have no parent or FGCHILD_IGNORE is set, you can't possibly recieve MOUSEMOVE events so don't bother with this.
  AbsRect out;
  ResolveRect(self, &out);
  if(HitAbsRect(&out, (FABS)fgroot_instance->mouse.x, (FABS)fgroot_instance->mouse.y))
    fgroot_instance->mouse.state |= FGMOUSE_SEND_MOUSEMOVE;
}

void FG_FASTCALL fgChild_AddListener(fgChild* self, unsigned short type, FN_LISTENER listener)
{
  fgListenerList.Insert(std::pair<fgChild*, unsigned short>(self, type));
  fgListenerHash.Insert(std::pair<fgChild*, unsigned short>(self, type), listener);
}

bss_util::AVL_Node<std::pair<fgChild*, unsigned short>>* FG_FASTCALL fgChild_GetAnyListener(fgChild* key, bss_util::AVL_Node<std::pair<fgChild*, unsigned short>>* cur)
{
  while(cur)
  {
    switch(SGNCOMPARE(cur->_key.first, key)) // by only comparing the first half of the key, we can find all the nodes for a given fgChild*
    {
    case -1: cur = cur->_left; break;
    case 1: cur = cur->_right; break;
    default: return cur;
    }
  }

  return 0;
}
void FG_FASTCALL fgChild_ClearListeners(fgChild* self)
{
  bss_util::AVL_Node<std::pair<fgChild*, unsigned short>>* cur;

  while(cur = fgChild_GetAnyListener(self, fgListenerList.GetRoot()))
  {
    fgListenerHash.Remove(cur->_key);
    fgListenerList.Remove(cur->_key);
  }
}

void fgChild::Construct()
{
  fgSendMsg<FG_CONSTRUCT>(this);
}
void FG_FASTCALL fgChild::Move(unsigned char subtype, fgChild* child, unsigned long long diff)
{
  fgSendSubMsg<FG_MOVE, void*, size_t>(this, subtype, child, diff);
}
size_t FG_FASTCALL fgChild::SetAlpha(float alpha)
{
  return fgSendMsg<FG_SETALPHA, float>(this, alpha);
}
size_t FG_FASTCALL fgChild::SetArea(const CRect& area)
{
  return fgSendMsg<FG_SETAREA, const void*>(this, &area);
}
size_t FG_FASTCALL fgChild::SetElement(const fgElement& element)
{
  return fgSendMsg<FG_SETELEMENT, const void*>(this, &element);
}
void FG_FASTCALL fgChild::SetFlag(fgFlag flag, bool value)
{
  fgSendMsg<FG_SETFLAG, ptrdiff_t, size_t>(this, flag, value != 0);
}
void FG_FASTCALL fgChild::SetFlags(fgFlag flags)
{
  fgSendMsg<FG_SETFLAGS, ptrdiff_t>(this, flags);
}
size_t FG_FASTCALL fgChild::SetMargin(const AbsRect& margin)
{
  return fgSendMsg<FG_SETMARGIN, const void*>(this, &margin);
}
size_t FG_FASTCALL fgChild::SetPadding(const AbsRect& padding)
{
  return fgSendMsg<FG_SETPADDING, const void*>(this, &padding);
}
void FG_FASTCALL fgChild::SetParent(fgChild* parent, fgChild* prev)
{
  fgSendMsg<FG_SETPARENT, void*, void*>(this, parent, prev);
}
size_t FG_FASTCALL fgChild::AddChild(fgChild* child)
{
  return fgSendMsg<FG_ADDCHILD, void*>(this, child);
}
size_t FG_FASTCALL fgChild::RemoveChild(fgChild* child)
{
  return fgSendMsg<FG_REMOVECHILD, void*>(this, child);
}
size_t FG_FASTCALL fgChild::LayoutFunction(const FG_Msg& msg, const CRect& area)
{
  return fgSendMsg<FG_LAYOUTFUNCTION, const void*, const void*>(this, &msg, &area);
}
void fgChild::LayoutChange(unsigned char subtype, fgChild* target, fgChild* old)
{
  fgSendSubMsg<FG_LAYOUTCHANGE, void*, void*>(this, subtype, target, old);
}
ptrdiff_t FG_FASTCALL fgChild::LayoutIndex(fgChild* child)
{
  return fgSendMsg<FG_LAYOUTINDEX, void*>(this, child);
}
size_t FG_FASTCALL fgChild::LayoutLoad(fgLayout* layout, FN_MAPPING mapping)
{
  return fgSendMsg<FG_LAYOUTLOAD, void*, void*>(this, layout, mapping);
}
size_t fgChild::Drag(fgChild* target, const FG_Msg& msg)
{
  return fgSendMsg<FG_DRAG, void*, const void*>(this, target, &msg);
}
size_t fgChild::Dragging(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_DRAGGING;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t fgChild::Drop(struct _FG_CHILD* target)
{
  return fgSendMsg<FG_DROP, void*>(this, target);
}
void fgChild::Draw(AbsRect* area, int dpi)
{
  fgSendMsg<FG_DRAW, void*, size_t>(this, area, dpi);
}
fgChild* FG_FASTCALL fgChild::Clone(fgChild* from)
{
  return reinterpret_cast<fgChild*>(fgSendMsg<FG_CLONE, void*>(this, from));
}
size_t FG_FASTCALL fgChild::SetSkin(fgSkin* skin, FN_MAPPING mapping)
{
  return fgSendMsg<FG_SETSKIN, void*, void*>(this, skin, mapping);
}
fgSkin* FG_FASTCALL fgChild::GetSkin(fgChild* child)
{
  return reinterpret_cast<fgSkin*>(fgSendMsg<FG_GETSKIN, void*>(this, child));
}
size_t FG_FASTCALL fgChild::SetStyle(const char* name)
{
  return fgSendMsg<FG_SETSTYLE, const void*, size_t>(this, name, 0);
}
size_t FG_FASTCALL fgChild::SetStyle(struct _FG_STYLE* style)
{
  return fgSendMsg<FG_SETSTYLE, void*, size_t>(this, style, 2);
}
size_t FG_FASTCALL fgChild::SetStyle(size_t index)
{
  return fgSendMsg<FG_SETSTYLE, ptrdiff_t, size_t>(this, index, 1);
}
struct _FG_STYLE* fgChild::GetStyle()
{
  return reinterpret_cast<struct _FG_STYLE*>(fgSendMsg<FG_GETSTYLE>(this));
}
const char* fgChild::GetClassName()
{
  return reinterpret_cast<const char*>(fgSendMsg<FG_GETCLASSNAME>(this));
}
size_t FG_FASTCALL fgChild::MouseDown(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEDOWN;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEDBLCLICK;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseUp(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEUP;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseOn(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEON;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseOff(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEOFF;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseMove(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEMOVE;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseScroll(int x, int y, unsigned short delta)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSESCROLL;
  m.x = x;
  m.y = y;
  m.scrolldelta = delta;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::MouseLeave(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSELEAVE;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::KeyUp(unsigned char keycode, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYUP;
  m.keycode = keycode;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::KeyDown(unsigned char keycode, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYDOWN;
  m.keycode = keycode;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::KeyChar(int keychar, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYCHAR;
  m.keychar = keychar;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::JoyButtonDown(short joybutton)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYBUTTONDOWN;
  m.joybutton = joybutton;
  m.joydown = true;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::JoyButtonUp(short joybutton)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYBUTTONUP;
  m.joybutton = joybutton;
  m.joydown = false;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t FG_FASTCALL fgChild::JoyAxis(float joyvalue, short joyaxis)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYAXIS;
  m.joyvalue = joyvalue;
  m.joyaxis = joyaxis;
  return (*fgroot_instance->behaviorhook)(this, &m);
}
size_t fgChild::GotFocus()
{
  return fgSendMsg<FG_GOTFOCUS>(this);
}
void fgChild::LostFocus()
{
  fgSendMsg<FG_LOSTFOCUS>(this);
}
size_t FG_FASTCALL fgChild::SetName(const char* name)
{
  return fgSendMsg<FG_SETNAME, const void*>(this, name);
}
const char* fgChild::GetName()
{
  return reinterpret_cast<const char*>(fgSendMsg<FG_GETNAME>(this));
}
void fgChild::Nuetral()
{
  fgSendMsg<FG_NUETRAL>(this);
}
void fgChild::Hover()
{
  fgSendMsg<FG_HOVER>(this);
}
void fgChild::Active()
{
  fgSendMsg<FG_ACTIVE>(this);
}
void fgChild::Action()
{
  fgSendMsg<FG_ACTION>(this);
}
fgChild* fgChild::GetSelectedItem()
{
  return reinterpret_cast<fgChild*>(fgSendMsg<FG_GETSELECTEDITEM>(this));
}
size_t FG_FASTCALL fgChild::SetResource(void* res)
{
  return fgSendMsg<FG_SETRESOURCE, void*>(this, res);
}
size_t FG_FASTCALL fgChild::SetUV(const CRect& uv)
{
  return fgSendMsg<FG_SETUV, const void*>(this, &uv);
}
size_t FG_FASTCALL fgChild::SetColor(unsigned int color, int index)
{
  return fgSendMsg<FG_SETCOLOR, ptrdiff_t, size_t>(this, color, index);
}
size_t FG_FASTCALL fgChild::SetOutline(float outline)
{
  return fgSendMsg<FG_SETOUTLINE, float>(this, outline);
}
size_t FG_FASTCALL fgChild::SetFont(void* font)
{
  return fgSendMsg<FG_SETFONT, void*>(this, font);
}
size_t FG_FASTCALL fgChild::SetText(const char* text)
{
  return fgSendMsg<FG_SETTEXT, const void*>(this, text);
}
void* fgChild::GetResource()
{
  return reinterpret_cast<void*>(fgSendMsg<FG_GETRESOURCE>(this));
}
const CRect* fgChild::GetUV()
{
  return reinterpret_cast<const CRect*>(fgSendMsg<FG_GETUV>(this));
}
unsigned int FG_FASTCALL fgChild::GetColor(int index)
{
  return (unsigned int)fgSendMsg<FG_GETCOLOR, ptrdiff_t>(this, index);
}
float fgChild::GetOutline()
{
  return *reinterpret_cast<float*>(fgSendMsg<FG_GETOUTLINE>(this));
}
void* fgChild::GetFont()
{
  return reinterpret_cast<void*>(fgSendMsg<FG_GETFONT>(this));
}
const char* fgChild::GetText()
{
  return reinterpret_cast<const char*>(fgSendMsg<FG_GETTEXT>(this));
}

void fgChild::AddListener(unsigned short type, FN_LISTENER listener)
{
  fgChild_AddListener(this, type, listener);
}