// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgBox.h"
#include "fgRoot.h"
#include "bss-util\bss_algo.h"
#include "feathercpp.h"

void FG_FASTCALL fgBox_Init(fgBox* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgBox_Destroy, (fgMessage)&fgBox_Message);
}
void FG_FASTCALL fgBox_Destroy(fgBox* self)
{
  ((bss_util::cDynArray<fgElement*>&)self->ordered).~cDynArray();
  fgScrollbar_Destroy(&self->scroll); // this will destroy our prechildren for us.
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
template<> BSS_FORCEINLINE char fgBoxVecCompare<FGBOX_TILE | FGBOX_DISTRIBUTEY>(const AbsVec& l, const AbsVec& r) { char ret = SGNCOMPARE(l.x, r.x); return !ret ? SGNCOMPARE(l.y, r.y) : ret; }

template<fgFlag FLAGS>
char fgBoxCompare(const AbsRect& l, const fgElement* const& e)
{
  AbsRect r;
  ResolveRect(e, &r);
  return fgBoxVecCompare<FLAGS&(FGBOX_TILE | FGBOX_DISTRIBUTEY)>(l.topleft, r.bottomright);
}
template<fgFlag FLAGS>
inline fgElement* fgBoxOrder(fgElement* self, const AbsRect* area)
{
  fgBox* box = (fgBox*)self;
  size_t r = bss_util::binsearch_near<const fgElement*, AbsRect, size_t, &fgBoxCompare<FLAGS>, &bss_util::CompT_EQ<char>, 1>(box->ordered.p, *area, 0, box->ordered.l);
  return (r >= box->ordered.l) ? box->ordered.p[0] : box->ordered.p[r];
}

template<fgFlag FLAGS> BSS_FORCEINLINE char fgBoxMsgCompare(const FG_Msg& l, const AbsRect& r);
template<> BSS_FORCEINLINE char fgBoxMsgCompare<FGBOX_TILEX>(const FG_Msg& l, const AbsRect& r) { return (l.x >= r.left) - (l.x < r.right); }
template<> BSS_FORCEINLINE char fgBoxMsgCompare<FGBOX_TILEY>(const FG_Msg& l, const AbsRect& r) { return (l.y >= r.top) - (l.y < r.bottom); }
template<> BSS_FORCEINLINE char fgBoxMsgCompare<FGBOX_TILE>(const FG_Msg& l, const AbsRect& r) { char ret = (l.y >= r.top) - (l.y < r.bottom); return !ret ? ((l.x >= r.left) - (l.x < r.right)) : ret; }
template<> BSS_FORCEINLINE char fgBoxMsgCompare<FGBOX_TILE | FGBOX_DISTRIBUTEY>(const FG_Msg& l, const AbsRect& r) { char ret = (l.x >= r.left) - (l.x < r.right); return !ret ? ((l.y >= r.top) - (l.y < r.bottom)) : ret; }

template<fgFlag FLAGS>
char fgBoxMsgCompare(const FG_Msg& l, const fgElement* const& e)
{
  AbsRect r;
  ResolveRect(e, &r);
  return fgBoxMsgCompare<FLAGS&(FGBOX_TILE | FGBOX_DISTRIBUTEY)>(l, r);
}
template<fgFlag FLAGS>
inline fgElement* fgBoxOrderInject(fgElement* self, const FG_Msg* msg)
{
  fgBox* box = (fgBox*)self;
  size_t r = bss_util::binsearch_near<const fgElement*, FG_Msg, size_t, &fgBoxMsgCompare<FLAGS>, &bss_util::CompT_EQ<char>, 1>(box->ordered.p, *msg, 0, box->ordered.l);
  return (r >= box->ordered.l) ? box->ordered.p[0] : box->ordered.p[r];
}

size_t FG_FASTCALL fgBox_Message(fgBox* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->scroll.control.element.flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    memset(&self->ordered, 0, sizeof(fgVectorElement));
    self->isordered = 1;
    self->fixedsize.x = -1;
    self->fixedsize.y = -1;
    break;
  case FG_SETFLAG: // Do the same thing fgElement does to resolve a SETFLAG into SETFLAGS
    otherint = T_SETBIT(flags, otherint, msg->otheraux);
  case FG_SETFLAGS:
    if((otherint^flags) & FGBOX_LAYOUTMASK)
    { // handle a layout flag change
      size_t r = fgScrollbar_Message(&self->scroll, msg); // we have to actually set the flags first before resetting the layout
      fgSubMessage(*self, FG_LAYOUTCHANGE, FGELEMENT_LAYOUTRESET, 0, 0);
      return r;
    }
    break;
  case FG_LAYOUTFUNCTION:
    if(self->scroll->flags&(FGBOX_TILEX | FGBOX_TILEY)) // TILE flags override DISTRIBUTE flags, if they're specified.
      return fgTileLayout(*self, (const FG_Msg*)msg->other, self->scroll->flags&FGBOX_LAYOUTMASK, (AbsVec*)msg->other2);
    if(self->scroll->flags&(FGBOX_DISTRIBUTEX | FGBOX_DISTRIBUTEY))
      return fgDistributeLayout(*self, (const FG_Msg*)msg->other, self->scroll->flags&FGBOX_LAYOUTMASK);
    break; // If no layout flags are specified, fall back to default layout behavior.
  case FG_REMOVECHILD:
  {
    fgElement* hold = (fgElement*)msg->other;
    for(size_t i = 0; i < self->ordered.l; ++i)
      if(self->ordered.p[i] == hold)
        ((bss_util::cDynArray<fgElement*>&)self->ordered).Remove(i);

    size_t r = fgScrollbar_Message(&self->scroll, msg);
    self->isordered = checkIsOrdered(self->scroll->root);
    return r;
  }
  case FG_ADDCHILD:
    assert(msg->other != 0);
    if(fgScrollbar_Message(&self->scroll, msg) == FG_ACCEPT)
    {
      fgElement* hold = (fgElement*)msg->other;
      fgElement* next = hold->next;
      if((hold->flags & (FGELEMENT_BACKGROUND | FGELEMENT_NOCLIP)) == FGELEMENT_NOCLIP)
        self->isordered = 0; // If ANY foreground elements are nonclipping, we can't use ordered rendering, because this would require us to maintain a second "nonclipping" sorted array.
      if(self->isordered)
      {
        fgElement* prev = !next ? self->scroll->last : next->prev;
        if(hold->flags&FGELEMENT_BACKGROUND) // If we're a background element, just make sure we aren't surrounded by foreground elements
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

      if(!(hold->flags&FGELEMENT_BACKGROUND))
      {
        if(self->isordered) // if we're still ordered then we add this to our vector
        {
          if(!next || next->flags&FGELEMENT_BACKGROUND)
            ((bss_util::cDynArray<fgElement*>&)self->ordered).Add(hold);
          else
          {
            size_t i = self->ordered.l;
            while(i > 0 && self->ordered.p[--i] != next);
            assert(self->ordered.p[i] == next);
            ((bss_util::cDynArray<fgElement*>&)self->ordered).Insert(hold, i);
          }
        }
        else
          self->ordered.l = 0;
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_DRAW:
    if(!self->isordered || !self->ordered.l)
      fgStandardDraw(*self, (AbsRect*)msg->other, msg->otheraux, msg->subtype & 1);
    else
    {
      fgElement* (*fn)(fgElement*, const AbsRect*);
      switch(self->scroll->flags&(FGBOX_TILE | FGBOX_DISTRIBUTEY))
      {
      case 0:
      case FGBOX_TILEX: fn = &fgBoxOrder<FGBOX_TILEX>; break;
      case FGBOX_TILEY: fn = &fgBoxOrder<FGBOX_TILEY>; break;
      case FGBOX_TILE: fn = &fgBoxOrder<FGBOX_TILE>; break;
      case FGBOX_TILE | FGBOX_DISTRIBUTEY: fn = &fgBoxOrder<FGBOX_TILE | FGBOX_DISTRIBUTEY>; break;
      }
      fgOrderedDraw(*self, (AbsRect*)msg->other, msg->otheraux, msg->subtype & 1, self->ordered.p[self->ordered.l - 1]->next, fn);
    }
    return FG_ACCEPT;
  case FG_INJECT:
    if(!self->isordered || !self->ordered.l)
      return fgStandardInject(*self, (const FG_Msg*)msg->other, (const AbsRect*)msg->other2);
    else
    {
      fgElement* (*fn)(fgElement*, const FG_Msg*);
      switch(self->scroll->flags&(FGBOX_TILE | FGBOX_DISTRIBUTEY))
      {
      case 0:
      case FGBOX_TILEX: fn = &fgBoxOrderInject<FGBOX_TILEX>; break;
      case FGBOX_TILEY: fn = &fgBoxOrderInject<FGBOX_TILEY>; break;
      case FGBOX_TILE: fn = &fgBoxOrderInject<FGBOX_TILE>; break;
      case FGBOX_TILE | FGBOX_DISTRIBUTEY: fn = &fgBoxOrderInject<FGBOX_TILE | FGBOX_DISTRIBUTEY>; break;
      }
      return fgOrderedInject(*self, (const FG_Msg*)msg->other, (const AbsRect*)msg->other2, self->ordered.p[self->ordered.l - 1]->next, fn);
    }
  case FG_GETCLASSNAME:
    return (size_t)"Box";
  }

  return fgScrollbar_Message(&self->scroll, msg);
}