// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgElement.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include "feathercpp.h"
#include <math.h>
#include <limits.h>

template<typename U, typename V>
BSS_FORCEINLINE char CompPairInOrder(const std::pair<U, V>& l, const std::pair<U, V>& r) { char ret = SGNCOMPARE(l.first, r.first); return !ret ? SGNCOMPARE(l.second, r.second) : ret; }

typedef bss_util::cDynArray<fgElement*, FG_UINT> fgSkinRefArray;
bss_util::cAVLtree<std::pair<fgElement*, unsigned short>, void, &CompPairInOrder> fgListenerList;

void FG_FASTCALL fgElement_InternalSetup(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, void (FG_FASTCALL *destroy)(void*), size_t(FG_FASTCALL *message)(void*, const FG_Msg*))
{
  assert(self != 0);
  memset(self, 0, sizeof(fgElement));
  self->destroy = destroy;
  self->free = 0;
  self->name = fgCopyText(name);
  self->message = message;
  self->flags = flags;
  self->style = (FG_UINT)-1;
  if(transform) self->transform = *transform;
  _sendmsg<FG_CONSTRUCT>(self);
  _sendmsg<FG_SETPARENT, void*, void*>(self, parent, next);
}

void FG_FASTCALL fgElement_Init(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(self, parent, next, name, flags, transform, (FN_DESTROY)&fgElement_Destroy, (FN_MESSAGE)&fgElement_Message);
}

void FG_FASTCALL fgElement_RemoveParent(fgElement* BSS_RESTRICT self)
{
  if(self->parent != 0)
  {
    fgSubMessage(self->parent, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTREMOVE, self, 0);
    if(self->parent->lastfocus == self)
      self->parent->lastfocus = 0;
    LList_RemoveAll(self); // Remove ourselves from our parent
  }
  self->next = 0;
  self->prev = 0;
}

void FG_FASTCALL fgElement_Destroy(fgElement* self)
{
  assert(self != 0);
  if(fgFocusedWindow == self)
    _sendmsg<FG_LOSTFOCUS>(self);
  if(fgLastHover == self)
    fgLastHover = 0;
  if(fgCaptureWindow == self)
    fgCaptureWindow = 0;

  if(self->name) free(self->name);
  fgElement_Clear(self);
  fgElement_RemoveParent(self);
  self->parent = 0;
  reinterpret_cast<fgSkinRefArray&>(self->skinrefs).~cDynArray();
  fgElement_ClearListeners(self);
  assert(fgFocusedWindow != self); // If these assertions fail something is wrong with how the message chain is constructed
  assert(fgLastHover != self);
  assert(fgCaptureWindow != self);
}

// (1<<1) resize x (2)
// (1<<2) resize y (4)
// (1<<3) move x (8)
// (1<<4) move y (16)

char FG_FASTCALL fgElement_PotentialResize(fgElement* self)
{
  return ((self->transform.area.left.rel != 0 || self->transform.area.right.rel != 0) << 3) // If you have nonzero relative coordinates, a resize will cause a move
    | ((self->transform.area.top.rel != 0 || self->transform.area.bottom.rel != 0) << 4)
    | ((self->transform.area.left.rel != self->transform.area.right.rel) << 1) // If you have DIFFERENT relative coordinates, a resize will cause a move AND a resize for you.
    | ((self->transform.area.top.rel != self->transform.area.bottom.rel) << 2);
}

fgElement* FG_FASTCALL fgElement_LoadLayout(fgElement* parent, fgElement* next, fgClassLayout* layout, FN_MAPPING mapping)
{
  fgElement* element = (*mapping)(layout->style.type, parent, next, layout->style.name, layout->style.flags, &layout->style.transform);
  fgVoidMessage(element, FG_SETSTYLE, &layout->style.style, 2);

  fgElement* p = 0;
  for(FG_UINT i = 0; i < layout->children.l; ++i)
    p = fgElement_LoadLayout(element, p, layout->children.p + i, mapping);

  return element;
}

size_t FG_FASTCALL fgElement_Message(fgElement* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgElement* hold;
  assert(self != 0);
  assert(msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    return 1;
  case FG_MOVE:
    if(!msg->other && self->parent != 0) // This is internal, so we must always propagate it up
      fgSubMessage(self->parent, FG_MOVE, msg->subtype, self, msg->otheraux | 1);
    if(msg->otheraux & 1) // A child moved, so recalculate any layouts
      fgSubMessage(self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTMOVE, msg->other, msg->otheraux);
    else if(msg->otheraux) // This was either internal or propagated down, in which case we must keep propagating it down so long as something changed.
    {
      fgElement* ref = !msg->other ? self : (fgElement*)msg->other;
      fgElement* cur = self->root;
      char diff;
      while(hold = cur)
      {
        cur = cur->next;
        diff = fgElement_PotentialResize(hold);

        fgSubMessage(hold, FG_MOVE, msg->subtype, ref, diff & msg->otheraux);
      }

      if(msg->otheraux & 0b100000110) // a layout change can happen on a resize or padding change
        fgSubMessage(self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTRESIZE, 0, msg->otheraux);
    }
    return 1;
  case FG_SETALPHA:
    return 0;
  case FG_SETAREA:
    if(!msg->other)
      return 0;
    {
      CRect* area = (CRect*)msg->other;
      char diff = CompareCRects(&self->transform.area, area);
      if(diff)
      {
        fgElement_MouseMoveCheck(self);
        fgDirtyElement(&self->transform);
        memcpy(&self->transform.area, area, sizeof(CRect));
        fgDirtyElement(&self->transform);
        fgElement_MouseMoveCheck(self);

        fgSubMessage(self, FG_MOVE, FG_SETAREA, 0, diff);
      }
    }
    return 1;
  case FG_SETTRANSFORM:
    if(!msg->other)
      return 0;
    {
      fgTransform* transform = (fgTransform*)msg->other;
      char diff = CompareTransforms(&self->transform, transform);
      if(diff)
      {
        fgElement_MouseMoveCheck(self);
        fgDirtyElement(&self->transform);
        memcpy(&self->transform, transform, sizeof(fgTransform));
        fgDirtyElement(&self->transform);
        fgElement_MouseMoveCheck(self);

        fgSubMessage(self, FG_MOVE, FG_SETTRANSFORM, 0, diff);
      }
    }
    return 1;
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = T_SETBIT(self->flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
  {
    fgFlag oldflags = self->flags;
    fgFlag change = self->flags ^ (fgFlag)otherint;
    self->flags = (fgFlag)otherint;
    if(change&FGELEMENT_IGNORE || change&FGELEMENT_NOCLIP)
    {
      self->flags = oldflags;
      fgElement* old = self->prev;
      LList_RemoveAll(self);
      self->flags = (fgFlag)otherint;
      LList_InsertAll(self, old);
    }
    if(change&FGELEMENT_BACKGROUND && self->parent != 0) // If we change the background we either have to add or remove this from the layout
      fgSubMessage(self->parent, FG_LAYOUTCHANGE, (self->flags & FGELEMENT_BACKGROUND) ? FGELEMENT_LAYOUTADD : FGELEMENT_LAYOUTREMOVE, self, 0);
    if((change&FGELEMENT_EXPAND)&self->flags) // If we change the expansion flags, we must recalculate every single child in our layout provided one of the expansion flags is actually set
      fgSubMessage(self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTRESET, 0, 0);
    if(change&FGELEMENT_HIDDEN || change&FGELEMENT_NOCLIP)
      fgDirtyElement(&self->transform);
  }
  return 1;
  case FG_SETMARGIN:
    if(!msg->other)
      return 0;
    {
      AbsRect* margin = (AbsRect*)msg->other;
      char diff = CompareMargins(&self->margin, margin);

      if(diff)
      {
        fgElement_MouseMoveCheck(self);
        fgDirtyElement(&self->transform);
        memcpy(&self->margin, margin, sizeof(AbsRect));
        fgDirtyElement(&self->transform);
        fgElement_MouseMoveCheck(self);

        fgSubMessage(self, FG_MOVE, FG_SETMARGIN, 0, diff);
      }
    }
    return 1;
  case FG_SETPADDING:
    if(!msg->other)
      return 0;
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareMargins(&self->padding, padding);

      if(diff)
      {
        fgElement_MouseMoveCheck(self);
        fgDirtyElement(&self->transform);
        memcpy(&self->padding, padding, sizeof(AbsRect));
        fgDirtyElement(&self->transform);
        fgElement_MouseMoveCheck(self);

        fgSubMessage(self, FG_MOVE, FG_SETPADDING, 0, diff | (1 << 8));
      }
    }
    return 1;
  case FG_SETPARENT:
  {
    fgElement* parent = (fgElement*)msg->other;
    fgElement* prev = (fgElement*)msg->other2;
    if(self->parent == parent)
    {
      if(self->prev != prev)
      {
        assert(self->parent != 0);
        assert(prev->parent == self->parent);
        fgElement* old = self->prev;
        LList_RemoveAll(self);
        LList_InsertAll(self, prev);
        fgSubMessage(self->parent, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTREORDER, self, (ptrdiff_t)old);
      }
      return 1;
    }
    fgElement_RemoveParent(self);
    self->parent = parent;

    if(parent)
    {
      LList_InsertAll(self, prev);
      _sendmsg<FG_SETSKIN>(self);
      fgSubMessage(parent, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTADD, self, 0);
    }
    else
      _sendmsg<FG_SETSKIN>(self);
  }
    fgSubMessage(self, FG_MOVE, FG_SETPARENT, 0, fgElement_PotentialResize(self));
    return 1;
  case FG_ADDITEM:
  case FG_ADDCHILD:
    hold = (fgElement*)msg->other;
    if(!hold)
      return 0;
    return _sendmsg<FG_SETPARENT, void*>(hold, self);
  case FG_REMOVECHILD:
    hold = (fgElement*)msg->other;
    if(!msg->other || hold->parent != self)
      return 0;
    return _sendmsg<FG_SETPARENT>((fgElement*)msg->other);
  case FG_LAYOUTCHANGE:
  {
    CRect area = self->transform.area;
    if(_sendmsg<FG_LAYOUTFUNCTION, const void*, void*>(self, msg, &area) != 0 || (msg->otheraux & (1 << 8)) != 0)
    {
      if(self->flags&FGELEMENT_EXPANDX)
        area.right.abs += self->padding.left + self->padding.right;
      if(self->flags&FGELEMENT_EXPANDY)
        area.bottom.abs += self->padding.top + self->padding.bottom;
      _sendmsg<FG_SETAREA, void*>(self, &area);
    }
    return 1;
  }
  case FG_LAYOUTFUNCTION:
  {
    AbsRect area;
    ResolveRect(self, &area);
    return fgLayout_Default(self, (const FG_Msg*)msg->other, (CRect*)msg->other2, &area);
  }
  case FG_LAYOUTLOAD:
  {
    fgLayout* layout = (fgLayout*)msg->other;
    if(!layout)
      return 0;

    FN_MAPPING mapping = (FN_MAPPING)msg->other2;
    if(!mapping) mapping = &fgLayoutLoadMapping;

    fgElement* next = self->last;
    for(FG_UINT i = 0; i < layout->layout.l; ++i)
      next = fgElement_LoadLayout(self, next, layout->layout.p + i, mapping);
  }
  return 1;
  case FG_CLONE:
  {
    hold = (fgElement*)msg->other;
    if(!hold)
      hold = (fgElement*)malloc(sizeof(fgElement));
    memcpy(hold, self, sizeof(fgElement));
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
    hold->name = fgCopyText(self->name);
    hold->SetParent(self->parent, self->prev);

    fgElement* cur = self->root;
    while(cur)
    {
      _sendmsg<FG_ADDCHILD, void*>(hold, (fgElement*)_sendmsg<FG_CLONE>(cur));
      cur = cur->next;
    }

    if(!msg->other)
      return (size_t)hold;
  }
  return 0;
  case FG_GETCLASSNAME:
    return (size_t)"fgElement";
  case FG_GETSKIN:
    if(!msg->other)
      return (!self->parent) ? reinterpret_cast<size_t>(self->skin) : _sendmsg<FG_GETSKIN, void*>(self->parent, self);
    if(self->skin != 0)
    {
      hold = (fgElement*)msg->other;
      const char* name = hold->GetName();
      if(name)
      {
        fgSkin* skin = fgSkin_GetSkin(self->skin, name);
        if(skin != 0)
          return reinterpret_cast<size_t>(skin);
      }
      name = hold->GetClassName();

      fgSkin* skin = fgSkin_GetSkin(self->skin, name);
      if(skin != 0)
        return reinterpret_cast<size_t>(skin);
    }
    return (!self->parent)? 0 : fgPassMessage(self->parent, msg);
  case FG_SETSKIN:
  {
    fgSkin* skin = (fgSkin*)(!msg->other ? (void*)_sendmsg<FG_GETSKIN>(self) : msg->other);
    if(self->skin != skin) // only bother changing the skin if there's stuff to change
    {
      if(self->skin != 0) // remove existing skin elements
      {
        for(FG_UINT i = 0; i < self->skinrefs.l; ++i)
          _sendmsg<FG_REMOVECHILD, void*>(self, self->skinrefs.p[i]);
      }
      self->skinrefs.l = 0;
      self->skin = skin;
      if(self->skin != 0)
      {
        FN_MAPPING mapping = (FN_MAPPING)msg->other2;
        if(!mapping) mapping = &fgLayoutLoadMapping;

        fgElement* child = self->last;
        for(FG_UINT i = 0; i < self->skin->children.l; ++i)
        {
          fgStyleLayout* layout = self->skin->children.p + i;
          child = (*mapping)(layout->type, self, child, layout->name, layout->flags, &layout->transform);
          fgVoidMessage(child, FG_SETSTYLE, &layout->style, 2);
          ((fgSkinRefArray&)self->skinrefs).Add(child);
        }
        fgVoidMessage(self, FG_SETSTYLE, (void*)&self->skin->style, 2);
      }
    }

    fgElement* cur = self->root;
    while(cur) // Because we allow children to look up their entire inheritance tree we must always inform them of skin changes
    {
      cur->SetSkin(0, (FN_MAPPING)msg->other2);
      cur = cur->next;
    }
    fgIntMessage(self, FG_SETSTYLE, -1, 1); // force us and our children to recalculate our style based on the new skin
  }
  return 1;
  case FG_SETSTYLE:
  {
    fgStyle* style = 0;
    if(msg->otheraux != 2)
    {
      size_t index = (!msg->otheraux ? fgStyle_GetName((const char*)msg->other) : (size_t)msg->otherint);

      if(index == -1)
        index = _sendmsg<FG_GETSTYLE>(self);
      else
        self->style = (FG_UINT)index;

      if(self->skin != 0 && index < self->skin->styles.l)
        style = self->skin->styles.p + index;

      fgElement* cur = self->root;
      while(cur)
      {
        fgIntMessage(cur, FG_SETSTYLE, -1, 1); // Forces the child to recalculate the style inheritance
        cur = cur->next;
      }
    }
    else
      style = (fgStyle*)msg->other;

    if(!style)
      return 0;

    fgStyleMsg* cur = style->styles;
    while(cur)
    {
      (*fgroot_instance->behaviorhook)(self, &cur->msg);
      cur = cur->next;
    }
  }
  return 1;
  case FG_GETSTYLE:
    return (self->style == (FG_UINT)-1 && self->parent != 0) ? _sendmsg<FG_GETSTYLE>(self->parent) : self->style;
  case FG_GOTFOCUS:
    if(self->lastfocus) {
      fgElement* hold = self->lastfocus;
      self->lastfocus = 0;
      if(_sendmsg<FG_GOTFOCUS>(hold))
        return 1;
    }
    if(self->parent)
      return _sendmsg<FG_GOTFOCUS>(self->parent);
    break;
  case FG_DRAG:
  {
    fgroot_instance->drag = (fgElement*)msg->other;
    FG_Msg m = *(FG_Msg*)msg->other2; // Immediately trigger an FG_DRAGGING message to set the cursor
    m.type = FG_DRAGGING;
    fgPassMessage(self, &m);
    return 1;
  }
  case FG_DRAGGING:
    fgSetCursor(FGCURSOR_NO, 0);
    return 0;
  case FG_DROP:
    fgSetCursor(FGCURSOR_ARROW, 0);
    return 0;
  case FG_DRAW:
    fgStandardDraw(self, (AbsRect*)msg->other, msg->otheraux, msg->subtype&1);
    return 1;
  case FG_SETNAME:
    if(self->name) free(self->name);
    self->name = fgCopyText((const char*)msg->other);
    _sendmsg<FG_SETSKIN, void*>(self, 0); // force the skin to be recalculated
    return 1;
  case FG_GETNAME:
    return (size_t)self->name;
  case FG_GETDPI:
    return self->parent ? _sendmsg<FG_GETDPI>(self->parent) : 0;
  case FG_GETLINEHEIGHT:
    return self->parent ? _sendmsg<FG_GETLINEHEIGHT>(self->parent) : 0;
  case FG_SETDPI:
    fgElement* cur = self->root;
    while(hold = cur)
    {
      cur = cur->next;
      fgPassMessage(hold, msg);
    }
    break;
  }

  return 0;
}

void FG_FASTCALL VirtualFreeChild(fgElement* self)
{
  assert(self != 0);
  (*self->destroy)(self);
  if(self->free)
    (*self->free)(self);
}

void FG_FASTCALL ResolveRect(const fgElement* self, AbsRect* out)
{
  assert(out != 0);
  if(!self->parent)
  {
    const CRect& v = self->transform.area;
    out->left = v.left.abs;
    out->top = v.top.abs;
    out->right = v.right.abs;
    out->bottom = v.bottom.abs;
    return;
  }

  AbsRect last;
  ResolveRect(self->parent, &last);
  ResolveRectCache(self, out, &last, (self->flags & FGELEMENT_BACKGROUND) ? 0 : &self->parent->padding);
}

void FG_FASTCALL ResolveRectCache(const fgElement* self, AbsRect* BSS_RESTRICT out, const AbsRect* BSS_RESTRICT last, const AbsRect* BSS_RESTRICT padding)
{
  AbsRect replace;
  if(padding != 0)
  {
    replace.left = last->left + padding->left;
    replace.top = last->top + padding->top;
    replace.right = last->right - padding->right;
    replace.bottom = last->bottom - padding->bottom;
    last = &replace;
  }

  AbsVec center = { self->transform.center.x.abs, self->transform.center.y.abs };
  const CRect* v = &self->transform.area;
  assert(out != 0 && self != 0 && last != 0);
  out->left = lerp(last->left, last->right, v->left.rel) + v->left.abs;
  out->top = lerp(last->top, last->bottom, v->top.rel) + v->top.abs;
  out->right = lerp(last->left, last->right, v->right.rel) + v->right.abs;
  out->bottom = lerp(last->top, last->bottom, v->bottom.rel) + v->bottom.abs;

  center.x += (v->right.abs - v->left.abs)*self->transform.center.x.rel;
  center.y += (v->bottom.abs - v->top.abs)*self->transform.center.y.rel;
  //center = ResolveVec(&,r); // We can't use this because the center is relative to the DIMENSIONS, not the actual position.
  out->left -= center.x;
  out->top -= center.y;
  out->right -= center.x;
  out->bottom -= center.y;
  out->left += self->margin.left;
  out->top += self->margin.top;
  out->right -= self->margin.right;
  out->bottom -= self->margin.bottom;

  if(self->flags&FGELEMENT_SNAPX) { // TODO: Make this work properly with DPI so it maps to pixels
    out->left = floor(out->left);
    out->right = floor(out->right);
  }
  if(self->flags&FGELEMENT_SNAPY) {
    out->top = floor(out->top);
    out->bottom = floor(out->bottom);
  }
}


char FG_FASTCALL MsgHitCRect(const FG_Msg* msg, const fgElement* child)
{
  AbsRect r;
  assert(msg != 0 && child != 0);
  ResolveRect(child, &r);
  return MsgHitAbsRect(msg, &r);
}

BSS_FORCEINLINE fgElement*& fgElement_prev(fgElement* p) { return p->prev; }
BSS_FORCEINLINE fgElement*& fgElement_prevclip(fgElement* p) { return p->prevclip; }

BSS_FORCEINLINE fgElement*& fgElement_next(fgElement* p) { return p->next; }
BSS_FORCEINLINE fgElement*& fgElement_nextclip(fgElement* p) { return p->nextclip; }

template<fgElement*&(*PREV)(fgElement*), fgElement*&(*NEXT)(fgElement*)>
void FG_FASTCALL LList_Remove(fgElement* self, fgElement** root, fgElement** last)
{
  assert(self != 0);
  if(PREV(self) != 0) NEXT(PREV(self)) = NEXT(self);
  else *root = NEXT(self);
  if(NEXT(self) != 0) PREV(NEXT(self)) = PREV(self);
  else *last = PREV(self);
}

void FG_FASTCALL LList_RemoveAll(fgElement* self)
{
  LList_Remove<fgElement_prev, fgElement_next>(self, &self->parent->root, &self->parent->last); // Remove ourselves from our parent
  if(!(self->flags&FGELEMENT_IGNORE))
  {
    fgElement_MouseMoveCheck(self); // we have to check if the mouse intersects the child BEFORE we remove it so we can properly resolve relative coordinates.
    if(self->flags&FGELEMENT_NOCLIP)
      LList_Remove<fgElement_prevclip, fgElement_nextclip>(self, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Remove<fgElement_prevclip, fgElement_nextclip>(self, &self->parent->rootclip, &self->parent->lastclip);
  }
}

template<fgElement*&(*PREV)(fgElement*), fgElement*&(*NEXT)(fgElement*)>
void FG_FASTCALL LList_Insert(fgElement* self, fgElement* cur, fgElement* prev, fgElement** root, fgElement** last)
{
  NEXT(self) = cur;
  PREV(self) = prev;
  if(prev) NEXT(prev) = self;
  else *root = self; // Prev is only null if we're inserting before the root, which means we must reassign the root.
  if(cur) PREV(cur) = self;
  else *last = self; // Cur is null if we are at the end of the list, so update last
}

void FG_FASTCALL LList_InsertAll(fgElement* BSS_RESTRICT self, fgElement* BSS_RESTRICT next)
{
  fgElement* prev = !next ? self->parent->last : next->prev;
  LList_Insert<fgElement_prev, fgElement_next>(self, next, prev, &self->parent->root, &self->parent->last);
  if(!(self->flags&FGELEMENT_IGNORE))
  {
    if(self->flags&FGELEMENT_NOCLIP)
      LList_Insert<fgElement_prevclip, fgElement_nextclip>(self, next, prev, &self->parent->rootnoclip, &self->parent->lastnoclip);
    else
      LList_Insert<fgElement_prevclip, fgElement_nextclip>(self, next, prev, &self->parent->rootclip, &self->parent->lastclip);
    fgElement_MouseMoveCheck(self); // This has to be AFTER so we can properly resolve relative coordinates
  }
}

size_t FG_FASTCALL fgVoidMessage(fgElement* self, unsigned char type, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

size_t FG_FASTCALL fgIntMessage(fgElement* self, unsigned char type, ptrdiff_t data, size_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.otherint = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}

FG_EXTERN size_t FG_FASTCALL fgPassMessage(fgElement* self, const FG_Msg* msg)
{
  return (*fgroot_instance->behaviorhook)(self, msg);
}

FG_EXTERN size_t FG_FASTCALL fgSubMessage(fgElement* self, unsigned char type, unsigned char subtype, void* data, ptrdiff_t aux)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = subtype;
  msg.other = data;
  msg.otheraux = aux;
  assert(self != 0);
  return (*fgroot_instance->behaviorhook)(self, &msg);
}


inline char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgElement* child, FABS dimx)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  FABS reld = (selfarea->right.rel - selfarea->left.rel)*dimx;
  FABS right = (area->right.rel - area->left.rel)*(d + reld);
  if(area->right.abs + right > d + reld)
  {
    FABS ndim = area->right.abs + right - reld;
    selfarea->right.abs = selfarea->left.abs + ndim;
    area->right.abs += (area->right.rel - area->left.rel)*(d - ndim);
    return 2;
  }
  return 0;
}

inline char FG_FASTCALL fgLayout_ExpandY(CRect* selfarea, fgElement* child, FABS dimy)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->bottom.abs - selfarea->top.abs;
  FABS reld = (selfarea->bottom.rel - selfarea->top.rel)*dimy;
  FABS bottom = (area->bottom.rel - area->top.rel)*(d + reld);
  if(area->bottom.abs + bottom > d + reld)
  {
    FABS ndim = area->bottom.abs + bottom - reld;
    selfarea->bottom.abs = selfarea->top.abs + ndim;
    area->bottom.abs += (area->bottom.rel - area->top.rel)*(d - ndim); // Equivelent to: area->bottom.rel*(d + reld) - area->bottom.rel*(ndim + reld)
    return 4;
  }
  return 0;
}

size_t FG_FASTCALL fgLayout_Default(fgElement* self, const FG_Msg* msg, CRect* area, AbsRect* parent)
{
  if(!(self->flags & FGELEMENT_EXPAND))
    return 0;

  fgElement* child = (fgElement*)msg->other;

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTREORDER:
  case FGELEMENT_LAYOUTRESIZE:
    break;
  case FGELEMENT_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGELEMENT_LAYOUTREMOVE;
    size_t dim = fgLayout_Default(self, &m, area, parent);
    m.subtype = FGELEMENT_LAYOUTADD;
    dim |= fgLayout_Default(self, &m, area, parent);
    return dim;
  }
  case FGELEMENT_LAYOUTADD:
  {
    if(child->flags & FGELEMENT_BACKGROUND)
      break;
    char dim = 0;
    if(self->flags & FGELEMENT_EXPANDX)
      dim |= fgLayout_ExpandX(area, child, parent->right - parent->left);
    if(self->flags & FGELEMENT_EXPANDY)
      dim |= fgLayout_ExpandY(area, child, parent->bottom - parent->top);
    return dim;
  }
  case FGELEMENT_LAYOUTREMOVE:
  {
    if(child->flags & FGELEMENT_BACKGROUND)
      break;
    CRect* childarea = &child->transform.area;
    FABS dx = self->transform.area.right.abs - self->transform.area.left.abs;
    FABS dy = self->transform.area.bottom.abs - self->transform.area.top.abs;
    if(((self->flags & FGELEMENT_EXPANDX) &&
      childarea->right.rel == childarea->left.rel &&
      childarea->right.abs >= dx) ||
      ((self->flags & FGELEMENT_EXPANDY) &&
        childarea->bottom.rel == childarea->top.rel &&
        childarea->bottom.abs >= dy))
    {
      FG_Msg m = *msg;
      m.subtype = FGELEMENT_LAYOUTRESET;
      m.other = child;
      return fgLayout_Default(self, &m, area, parent);
    }
  }
  case FGELEMENT_LAYOUTRESET:
  {
    fgElement* hold = self->root;
    if(self->flags & FGELEMENT_EXPANDX)
      area->right.abs = area->left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
    if(self->flags & FGELEMENT_EXPANDY)
      area->bottom.abs = area->top.abs;
    char dim = 0;

    while(hold) // O(n) by necessity
    {
      if(hold != child)
      {
        if(self->flags & FGELEMENT_EXPANDX)
          dim |= fgLayout_ExpandX(area, hold, parent->right - parent->left);
        if(self->flags & FGELEMENT_EXPANDY)
          dim |= fgLayout_ExpandY(area, hold, parent->bottom - parent->top);
      }
      hold = hold->next;
    }

    return dim;
  }
  }

  return 0;
}

char fgLayout_TileSame(FABS posx, FABS posy, fgElement* cur)
{
  return (posx > 0 || posy > 0) && posx == cur->transform.area.left.abs && posy == cur->transform.area.left.abs;
}

void fgLayout_TileReorder(fgElement* prev, fgElement* child, char axis, FABS max, FABS pitch) // axis: 0 means x-axis first, 1 means y-axis first
{
  AbsVec pos = { 0,0 };
  if(prev)
  {
    pos.x = axis ? prev->transform.area.bottom.abs : prev->transform.area.right.abs;
    pos.y = axis ? prev->transform.area.left.abs : prev->transform.area.top.abs;
  }
  fgElement* cur = child;
  char rowbump = 0; // Used to keep track if we've gone down at least one row - this is required for our same position optimization to be valid.

  while(cur) // O(n) complexity, but should break after re-ordering becomes unnecessary.
  {
    if(!(cur->flags&FGELEMENT_BACKGROUND))
    {
      if(fgLayout_TileSame(axis ? pos.y : pos.x, axis ? pos.x : pos.y, cur) && rowbump)
        break;
      AbsVec dim = { (cur->transform.area.right.abs - cur->transform.area.left.abs), (cur->transform.area.bottom.abs - cur->transform.area.top.abs) };
      if(axis) { FABS t = dim.x; dim.x = dim.y; dim.y = t; }

      if(pos.x + dim.x > max)
      {
        pos.y += pitch;
        pos.x = 0;
        pitch = 0;
        rowbump = 1;
      }

      if(!axis)
        MoveCRect(pos.x, pos.y, &cur->transform.area);
      else
        MoveCRect(pos.y, pos.x, &cur->transform.area);
      pos.x += dim.x;
      if(dim.y > pitch) pitch = dim.y;
    }
    cur = cur->next;
  }
}

FABS FG_FASTCALL fgLayout_TileGetPitch(fgElement* cur, char axis)
{
  if(!cur) return 0.0f;
  FABS start = (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs);
  FABS dim;
  FABS pitch = 0.0;
  while(cur != 0 && (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs) == start)
  {
    dim = (axis ? cur->transform.area.right.abs : cur->transform.area.bottom.abs) - start;
    if(dim > pitch)
      pitch = dim;
    cur = cur->prev; // run backwards until we go up a level
  }
  return pitch;
}

size_t FG_FASTCALL fgLayout_Tile(fgElement* self, const FG_Msg* msg, char axes)
{
  fgElement* child = (fgElement*)msg->other;
  char axis = ((axes & 3) && (axes & 8)) || ((axes & 2) && !(axes & 1));
  FABS max = INFINITY;
  if(axes & 3)
  {
    AbsRect out;
    ResolveRect(self, &out);
    if((axes & 8) && !(self->flags & FGELEMENT_EXPANDY)) max = out.bottom - out.top;
    else if(!(self->flags & FGELEMENT_EXPANDX)) max = out.right - out.left;
  }

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTMOVE:
    break;
  case FGELEMENT_LAYOUTRESIZE:
    if((((msg->otheraux & 0b10) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->otheraux & 0b100) && !(self->flags&FGELEMENT_EXPANDY))) && (axes & 3))
      fgLayout_TileReorder(0, self->root, axis, max, 0.0f);
    break;
  case FGELEMENT_LAYOUTREORDER:
  {
    fgElement* old = (fgElement*)msg->other2;
    fgElement* cur = old;
    while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
    child = !cur ? child : old;
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
  }
  break;
  case FGELEMENT_LAYOUTADD:
    fgLayout_TileReorder(child->prev, child, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  case FGELEMENT_LAYOUTREMOVE:
    fgLayout_TileReorder(child->prev, child->next, axis, max, (axes & 3) ? fgLayout_TileGetPitch(child->prev, axis) : 0.0f);
    break;
  }

  return 0;
}

size_t FG_FASTCALL fgLayout_Distribute(fgElement* self, const FG_Msg* msg, char axis)
{
  /*switch(msg->type)
  {
  case FGELEMENT_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGELEMENT_LAYOUTRESIZE:
  break;
  case FGELEMENT_LAYOUTREORDER:
  //child = child->order<msg->other->order?child:msg->other; // Get lowest child
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTADD:
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTREMOVE:
  fgLayout_DistributeReorder(child->prev, child->next, axis, num);
  break;
  }*/

  return 0;
}

void FG_FASTCALL fgElement_Clear(fgElement* self)
{
  while(self->root) // Destroy all children
    VirtualFreeChild(self->root);
}

fgElement* FG_FASTCALL fgElement_GetChildUnderMouse(fgElement* self, int x, int y, AbsRect* cache)
{
  ResolveRect(self, cache);
  AbsRect child;
  fgElement* cur = self->root; // we have to go through the whole child list because you can have FG_IGNORE on but still be a hover target, espiecally if you're an fgText control inside a list.
  while(cur != 0)
  {
    if(!(cur->flags&FGELEMENT_BACKGROUND))
    {
      ResolveRectCache(cur, &child, cache, &self->padding); // this is only done for non background elements, so we always pass in the padding.
      if(HitAbsRect(&child, (FABS)x, (FABS)y))
        return cur;
    }
    cur = cur->next;
  }
  return 0;
}

void FG_FASTCALL fgElement_MouseMoveCheck(fgElement* self)
{
  if(!self->parent || (self->flags&FGELEMENT_IGNORE))
    return; // If you have no parent or FGELEMENT_IGNORE is set, you can't possibly recieve MOUSEMOVE events so don't bother with this.
  AbsRect out;
  ResolveRect(self, &out);
  if(HitAbsRect(&out, (FABS)fgroot_instance->mouse.x, (FABS)fgroot_instance->mouse.y))
    fgroot_instance->mouse.state |= FGMOUSE_SEND_MOUSEMOVE;
}

void FG_FASTCALL fgElement_AddListener(fgElement* self, unsigned short type, FN_LISTENER listener)
{
  fgListenerList.Insert(std::pair<fgElement*, unsigned short>(self, type));
  fgListenerHash.Insert(std::pair<fgElement*, unsigned short>(self, type), listener);
}

bss_util::AVL_Node<std::pair<fgElement*, unsigned short>>* FG_FASTCALL fgElement_GetAnyListener(fgElement* key, bss_util::AVL_Node<std::pair<fgElement*, unsigned short>>* cur)
{
  while(cur)
  {
    switch(SGNCOMPARE(cur->_key.first, key)) // by only comparing the first half of the key, we can find all the nodes for a given fgElement*
    {
    case -1: cur = cur->_left; break;
    case 1: cur = cur->_right; break;
    default: return cur;
    }
  }

  return 0;
}
void FG_FASTCALL fgElement_ClearListeners(fgElement* self)
{
  bss_util::AVL_Node<std::pair<fgElement*, unsigned short>>* cur;

  while(cur = fgElement_GetAnyListener(self, fgListenerList.GetRoot()))
  {
    fgListenerHash.Remove(cur->_key);
    fgListenerList.Remove(cur->_key);
  }
}

void fgElement::Construct() { _sendmsg<FG_CONSTRUCT>(this); }

void FG_FASTCALL fgElement::Move(unsigned char subtype, fgElement* child, unsigned long long diff) { _sendsubmsg<FG_MOVE, void*, size_t>(this, subtype, child, diff); }

size_t FG_FASTCALL fgElement::SetAlpha(float alpha) { return _sendmsg<FG_SETALPHA, float>(this, alpha); }

size_t FG_FASTCALL fgElement::SetArea(const CRect& area) { return _sendmsg<FG_SETAREA, const void*>(this, &area); }

size_t FG_FASTCALL fgElement::SetElement(const fgTransform& element) { return _sendmsg<FG_SETTRANSFORM, const void*>(this, &element); }

void FG_FASTCALL fgElement::SetFlag(fgFlag flag, bool value) { _sendmsg<FG_SETFLAG, ptrdiff_t, size_t>(this, flag, value != 0); }

void FG_FASTCALL fgElement::SetFlags(fgFlag flags) { _sendmsg<FG_SETFLAGS, ptrdiff_t>(this, flags); }

size_t FG_FASTCALL fgElement::SetMargin(const AbsRect& margin) { return _sendmsg<FG_SETMARGIN, const void*>(this, &margin); }

size_t FG_FASTCALL fgElement::SetPadding(const AbsRect& padding) { return _sendmsg<FG_SETPADDING, const void*>(this, &padding); }

void FG_FASTCALL fgElement::SetParent(fgElement* parent, fgElement* next) { _sendmsg<FG_SETPARENT, void*, void*>(this, parent, next); }

size_t FG_FASTCALL fgElement::AddChild(fgElement* child) { return _sendmsg<FG_ADDCHILD, void*>(this, child); }

size_t FG_FASTCALL fgElement::RemoveChild(fgElement* child) { return _sendmsg<FG_REMOVECHILD, void*>(this, child); }

size_t FG_FASTCALL fgElement::LayoutFunction(const FG_Msg& msg, const CRect& area) { return _sendmsg<FG_LAYOUTFUNCTION, const void*, const void*>(this, &msg, &area); }

void fgElement::LayoutChange(unsigned char subtype, fgElement* target, fgElement* old) { _sendsubmsg<FG_LAYOUTCHANGE, void*, void*>(this, subtype, target, old); }

size_t FG_FASTCALL fgElement::LayoutLoad(fgLayout* layout, FN_MAPPING mapping) { return _sendmsg<FG_LAYOUTLOAD, void*, void*>(this, layout, mapping); }

size_t fgElement::Drag(fgElement* target, const FG_Msg& msg) { return _sendmsg<FG_DRAG, void*, const void*>(this, target, &msg); }

size_t fgElement::Dragging(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_DRAGGING;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t fgElement::Drop(struct _FG_ELEMENT* target) { return _sendmsg<FG_DROP, void*>(this, target); }

void fgElement::Draw(AbsRect* area, int dpi) { _sendmsg<FG_DRAW, void*, size_t>(this, area, dpi); }

fgElement* FG_FASTCALL fgElement::Clone(fgElement* from) { return reinterpret_cast<fgElement*>(_sendmsg<FG_CLONE, void*>(this, from)); }

size_t FG_FASTCALL fgElement::SetSkin(fgSkin* skin, FN_MAPPING mapping) { return _sendmsg<FG_SETSKIN, void*, void*>(this, skin, mapping); }

fgSkin* FG_FASTCALL fgElement::GetSkin(fgElement* child) { return reinterpret_cast<fgSkin*>(_sendmsg<FG_GETSKIN, void*>(this, child)); }

size_t FG_FASTCALL fgElement::SetStyle(const char* name) { return _sendmsg<FG_SETSTYLE, const void*, size_t>(this, name, 0); }

size_t FG_FASTCALL fgElement::SetStyle(struct _FG_STYLE* style) { return _sendmsg<FG_SETSTYLE, void*, size_t>(this, style, 2); }

size_t FG_FASTCALL fgElement::SetStyle(size_t index) { return _sendmsg<FG_SETSTYLE, ptrdiff_t, size_t>(this, index, 1); }

struct _FG_STYLE* fgElement::GetStyle() { return reinterpret_cast<struct _FG_STYLE*>(_sendmsg<FG_GETSTYLE>(this)); }

const char* fgElement::GetClassName() { return reinterpret_cast<const char*>(_sendmsg<FG_GETCLASSNAME>(this)); }

size_t FG_FASTCALL fgElement::MouseDown(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEDOWN;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseDblClick(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEDBLCLICK;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseUp(int x, int y, unsigned char button, unsigned char allbtn)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEUP;
  m.x = x;
  m.y = y;
  m.button = button;
  m.allbtn = allbtn;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseOn(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEON;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseOff(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEOFF;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseMove(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSEMOVE;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseScroll(int x, int y, unsigned short delta)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSESCROLL;
  m.x = x;
  m.y = y;
  m.scrolldelta = delta;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::MouseLeave(int x, int y)
{
  FG_Msg m = { 0 };
  m.type = FG_MOUSELEAVE;
  m.x = x;
  m.y = y;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::KeyUp(unsigned char keycode, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYUP;
  m.keycode = keycode;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::KeyDown(unsigned char keycode, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYDOWN;
  m.keycode = keycode;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::KeyChar(int keychar, char sigkeys)
{
  FG_Msg m = { 0 };
  m.type = FG_KEYCHAR;
  m.keychar = keychar;
  m.sigkeys = sigkeys;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::JoyButtonDown(short joybutton)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYBUTTONDOWN;
  m.joybutton = joybutton;
  m.joydown = true;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::JoyButtonUp(short joybutton)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYBUTTONUP;
  m.joybutton = joybutton;
  m.joydown = false;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t FG_FASTCALL fgElement::JoyAxis(float joyvalue, short joyaxis)
{
  FG_Msg m = { 0 };
  m.type = FG_JOYAXIS;
  m.joyvalue = joyvalue;
  m.joyaxis = joyaxis;
  return (*fgroot_instance->behaviorhook)(this, &m);
}

size_t fgElement::GotFocus() { return _sendmsg<FG_GOTFOCUS>(this); }

void fgElement::LostFocus() { _sendmsg<FG_LOSTFOCUS>(this); }

size_t FG_FASTCALL fgElement::SetName(const char* name) { return _sendmsg<FG_SETNAME, const void*>(this, name); }

const char* fgElement::GetName() { return reinterpret_cast<const char*>(_sendmsg<FG_GETNAME>(this)); }

void fgElement::Nuetral() { _sendmsg<FG_NUETRAL>(this); }

void fgElement::Hover() { _sendmsg<FG_HOVER>(this); }

void fgElement::Active() { _sendmsg<FG_ACTIVE>(this); }

void fgElement::Action() { _sendmsg<FG_ACTION>(this); }

size_t fgElement::GetState(ptrdiff_t aux) { return _sendmsg<FG_GETSTATE, ptrdiff_t>(this, aux); }

float fgElement::GetStatef(ptrdiff_t aux) { size_t r = _sendmsg<FG_GETSTATE, ptrdiff_t>(this, aux); return *reinterpret_cast<float*>(&r); }

size_t fgElement::SetState(ptrdiff_t state, size_t aux) { return _sendmsg<FG_SETSTATE, ptrdiff_t, size_t>(this, state, aux); }

size_t fgElement::SetStatef(float state, size_t aux) { return _sendmsg<FG_SETSTATE, float, size_t>(this, state, aux); }

fgElement* fgElement::GetSelectedItem() { return reinterpret_cast<fgElement*>(_sendmsg<FG_GETSELECTEDITEM>(this)); }

size_t FG_FASTCALL fgElement::SetResource(void* res) { return _sendmsg<FG_SETRESOURCE, void*>(this, res); }

size_t FG_FASTCALL fgElement::SetUV(const CRect& uv) { return _sendmsg<FG_SETUV, const void*>(this, &uv); }

size_t FG_FASTCALL fgElement::SetColor(unsigned int color, int index) { return _sendmsg<FG_SETCOLOR, ptrdiff_t, size_t>(this, color, index); }

size_t FG_FASTCALL fgElement::SetOutline(float outline) { return _sendmsg<FG_SETOUTLINE, float>(this, outline); }

size_t FG_FASTCALL fgElement::SetFont(void* font) { return _sendmsg<FG_SETFONT, void*>(this, font); }

size_t FG_FASTCALL fgElement::SetLineHeight(float lineheight) { return _sendmsg<FG_SETLINEHEIGHT, float>(this, lineheight); }

size_t FG_FASTCALL fgElement::SetLetterSpacing(float letterspacing) { return _sendmsg<FG_SETLETTERSPACING, float>(this, letterspacing); }

size_t FG_FASTCALL fgElement::SetText(const char* text) { return _sendmsg<FG_SETTEXT, const void*>(this, text); }

void* fgElement::GetResource() { return reinterpret_cast<void*>(_sendmsg<FG_GETRESOURCE>(this)); }

const CRect* fgElement::GetUV() { return reinterpret_cast<const CRect*>(_sendmsg<FG_GETUV>(this)); }

unsigned int FG_FASTCALL fgElement::GetColor(int index) { return (unsigned int)_sendmsg<FG_GETCOLOR, ptrdiff_t>(this, index); }

float fgElement::GetOutline() { return *reinterpret_cast<float*>(_sendmsg<FG_GETOUTLINE>(this)); }

void* fgElement::GetFont() { return reinterpret_cast<void*>(_sendmsg<FG_GETFONT>(this)); }

float fgElement::GetLineHeight() { size_t r = _sendmsg<FG_GETLINEHEIGHT>(this); return *reinterpret_cast<float*>(&r); }

float fgElement::GetLetterSpacing() { size_t r = _sendmsg<FG_GETLETTERSPACING>(this); return *reinterpret_cast<float*>(&r); }

const char* fgElement::GetText() { return reinterpret_cast<const char*>(_sendmsg<FG_GETTEXT>(this)); }

void fgElement::AddListener(unsigned short type, FN_LISTENER listener) { fgElement_AddListener(this, type, listener); }