// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (FN_DESTROY)&fgScrollbar_Destroy, (FN_MESSAGE)&fgScrollbar_Message);
}
void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self)
{
  fgControl_Destroy(&self->control);
}

void FG_FASTCALL fgScrollbar_Redim(fgScrollbar* self, CRect& area)
{
  if(self->barcache.x > 0.0) area.right.abs += self->barcache.x;
  if(self->barcache.y > 0.0) area.bottom.abs += self->barcache.y;
  if(!(self->control.element.flags&FGELEMENT_EXPANDX)) // If we actually are expanding along a given axis, use that result, otherwise replace it.
    area.right.abs = self->control.element.transform.area.right.abs;
  if(!(self->control.element.flags&FGELEMENT_EXPANDY))
    area.bottom.abs = self->control.element.transform.area.bottom.abs;
  _sendmsg<FG_SETAREA, void*>(*self, &area); // SETAREA will set MAXDIM appropriately
}

void FG_FASTCALL fgScrollbar_SetBars(fgScrollbar* self)
{
  AbsRect area;
  ResolveRect(*self, &area);
  if(self->barcache.x >= 0)
  {
    float dim = area.right - area.left - self->realpadding.left - self->realpadding.right;
    float length = self->realsize.x / dim;
    float pos = self->control.element.padding.left / (self->realsize.x - dim);
    self->btn[2]->SetArea(CRect { 0, (1.0f - length)*pos, 0, 0, 0, (1.0f - length)*pos + length, 0, 1.0f });
  }
  if(self->barcache.y >= 0)
  {
    float dim = area.bottom - area.top - self->realpadding.top - self->realpadding.bottom;
    float length = self->realsize.y / dim;
    float pos = self->control.element.padding.top / (self->realsize.y - dim);
    self->btn[5]->SetArea(CRect { 0, 0, 0, (1.0f - length)*pos, 0, 1.0f, 0, (1.0f - length)*pos + length });
  }
}

void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{
  AbsRect r;
  ResolveRect(*self, &r);
  AbsVec dim = { r.right - r.left, r.bottom - r.top };

  bool hideh = !!(self->control.element.flags&FGSCROLLBAR_HIDEH);
  bool hidev = !!(self->control.element.flags&FGSCROLLBAR_HIDEV);

  // We have to figure out which scrollbars are visible based on flags and our dimensions
  bool scrollx = !hideh && ((dim.x < self->realsize.x) || self->control.element.flags&FGSCROLLBAR_SHOWH);
  bool scrolly = !hidev && ((dim.y < self->realsize.y) || self->control.element.flags&FGSCROLLBAR_SHOWV);

  // If we are adding or removing a scrollbar, this will change the padding and could change everything else (e.g. for textboxes)
  if((self->barcache.x != 0.0f && scrollx ^ (self->barcache.x < 0)) || (self->barcache.y != 0.0f && scrolly ^ (self->barcache.y < 0)))
  {
    if(scrollx ^ (self->barcache.x < 0)) self->barcache.x = -self->barcache.x;
    if(scrolly ^ (self->barcache.y < 0)) self->barcache.y = -self->barcache.y;
    return fgScrollbar_Recalc(self);
  }
  
  // If we get this far, the padding has already been taken care of - append or remove scrollbars.
  _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->bg[0], 0, scrollx ? "visible" : "hidden", fgStyleGetMask("hidden", "visible"));
  _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->bg[1], 0, scrolly ? "visible" : "hidden", fgStyleGetMask("hidden", "visible"));

  // Set bar dimensions appropriately based on padding
  fgScrollbar_SetBars(self);
}

void FG_FASTCALL fgScrollbar_ApplyPadding(fgScrollbar* self, float x, float y)
{
  self->control.element.padding.left += x;
  self->control.element.padding.top += y;
  if(self->control.element.padding.left > self->realpadding.left)
    self->control.element.padding.left = self->realpadding.left;
  if(self->control.element.padding.top > self->realpadding.top)
    self->control.element.padding.top = self->realpadding.top;

  AbsRect r;
  ResolveRect(*self, &r);
  float minpaddingx = (r.right - r.left) - self->realsize.x;
  minpaddingx = bssmin(minpaddingx, 0);
  float minpaddingy = (r.bottom - r.top) - self->realsize.y;
  minpaddingy = bssmin(minpaddingy, 0);
  if(self->control.element.padding.left < minpaddingx)
    self->control.element.padding.left = minpaddingx;
  if(self->control.element.padding.top < minpaddingy)
    self->control.element.padding.top = minpaddingy;

  self->control.element.padding.right = self->realpadding.right + (self->barcache.x > 0) ? self->barcache.x : 0;
  self->control.element.padding.bottom = self->realpadding.bottom + (self->barcache.y > 0) ? self->barcache.y : 0;

  fgScrollbar_Recalc(self); // recalculate scrollbars from our new padding value
}

size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  static const float DEFAULT_LINEHEIGHT = 30.0f;
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    memset(&self->maxdim, 0, sizeof(CVec));
    memset(&self->realpadding, 0, sizeof(AbsRect));
    memset(&self->barcache, 0, sizeof(AbsVec));
    memset(&self->realsize, 0, sizeof(AbsVec));
    fgElement_Init(&self->bg[0], *self, 0, "fgScrollbar:horzbg", FGELEMENT_BACKGROUND | FGELEMENT_EXPANDY, &fgTransform_EMPTY);
    fgElement_Init(&self->bg[1], *self, 0, "fgScrollbar:vertbg", FGELEMENT_BACKGROUND | FGELEMENT_EXPANDX, &fgTransform_EMPTY);
    fgElement_Init(self->btn[0], *self, &self->bg[0], "fgScrollbar:scrollleft", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgElement_Init(self->btn[1], *self, &self->bg[0], "fgScrollbar:scrollright", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgElement_Init(self->btn[2], *self, &self->bg[0], "fgScrollbar:scrollhorz", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgElement_Init(self->btn[3], *self, &self->bg[1], "fgScrollbar:scrollup", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgElement_Init(self->btn[4], *self, &self->bg[1], "fgScrollbar:scrolldown", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgElement_Init(self->btn[5], *self, &self->bg[1], "fgScrollbar:scrollvert", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->bg[0], 0, "hidden", fgStyleGetMask("hidden", "visible"));
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->bg[1], 0, "hidden", fgStyleGetMask("hidden", "visible"));
    return FG_ACCEPT;
  case FG_MOVE:
    if(!msg->other && (msg->otheraux & FGMOVE_RESIZE)) // detect an INTERNAL area change (every other area change will be handled elsewhere)
      fgScrollbar_Recalc(self); // If we have an actual area change, the scrollbars need to be repositioned or possibly removed entirely.
    break;
  case FG_SETAREA: // The set area only tries to enforce maxdim on the ABS portion. Consequently, responding to dimension changes must be done in FG_MOVE
    if(msg->other != nullptr)
    {
      AbsRect r;
      ResolveRect(*self, &r);
      AbsVec d = ResolveVec(&self->maxdim, &r);
      CRect area = *(CRect*)msg->other;
      if(self->maxdim.x.abs >= 0 && area.right.abs - area.left.abs > d.x)
        area.right.abs = area.left.abs + d.x;
      if(self->maxdim.y.abs >= 0 && area.bottom.abs - area.top.abs > d.y)
        area.bottom.abs = area.top.abs + d.y;
      FG_Msg m = *msg;
      m.other = &area;
      return fgControl_HoverMessage(&self->control, &m);
    }
    return 0;
  case FG_SETPADDING:
    if(msg->other != nullptr)
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareMargins(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff) // Only send a move message if the change in padding could have resulted in an actual area change (this ensures a scroll delta change will NOT trigger an FG_MOVE)
        fgSubMessage(*self, FG_MOVE, FG_SETPADDING, 0, diff | FGMOVE_PADDING);
      else
        fgScrollbar_Recalc(self); // Recalculate scrollbar positions
      return FG_ACCEPT;
    }
    return 0;
  case FG_MOUSESCROLL:
  {
    FG_Msg m { 0 };
    m.type = FG_ACTION;
    m.subtype = FGSCROLLBAR_CHANGE;
    float lineheight = self->control.element.GetLineHeight();
    m.otherfaux = (msg->scrolldelta / 120.0f) * 3.0f * lineheight;
    m.otherf = (msg->scrollhdelta / 120.0f) * lineheight;

    fgPassMessage(*self, &m);
    return FG_ACCEPT;
  }
  case FG_LAYOUTCHANGE:
    {
      fgFlag flags = self->control.element.flags;
      auto area = self->control.element.transform.area;
      self->control.element.flags |= FGELEMENT_EXPAND;
      
      size_t dim = _sendmsg<FG_LAYOUTFUNCTION, const void*, void*>(*self, msg, &area);
      self->control.element.flags = flags;
      self->realsize = { area.right.abs - area.left.abs, area.bottom.abs - area.top.abs }; // retrieve real area and then reset to the area of the window.
      if(dim)
      {
        if(!(flags & FGELEMENT_EXPANDX))
        {
          area.left = self->control.element.transform.area.left;
          area.right = self->control.element.transform.area.right;
        }
        if(!(flags & FGELEMENT_EXPANDY))
        {
          area.top = self->control.element.transform.area.top;
          area.bottom = self->control.element.transform.area.bottom;
        }
        if(flags&FGELEMENT_EXPAND)
          self->control.element.SetArea(area);
        fgScrollbar_Redim(self, area);
      }
      return FG_ACCEPT;
    }
    break;
  case FG_GETLINEHEIGHT:
  {
    size_t r = fgControl_HoverMessage(&self->control, msg);
    return !r ? *reinterpret_cast<const size_t*>(&DEFAULT_LINEHEIGHT) : r;
  }
  case FG_ACTION:
    switch(msg->subtype)
    {
    case FGSCROLLBAR_PAGE: // By default a page scroll (clicking on the area outside of the bar or hitting pageup/pagedown) attempts to scroll by the width of the container in that direction.
    {
      AbsRect r;
      ResolveRect(*self, &r);
      fgScrollbar_ApplyPadding(self, (r.right - r.left)*(msg->otherint == 0 ? -1 : 1), (r.bottom - r.top)*(msg->otherint == 1 ? -1 : 1));
      return FG_ACCEPT;
    }
    case FGSCROLLBAR_BUTTON:
    {
      float x = msg->otherint < 2 ? -self->control.element.GetLineHeight() : self->control.element.GetLineHeight();
      if(msg->otherint % 2)
        fgScrollbar_ApplyPadding(self, x, 0);
      else
        fgScrollbar_ApplyPadding(self, 0, x);
      return FG_ACCEPT;
    }
    case FGSCROLLBAR_CHANGE: // corresponds to an actual change amount. First argument is x axis, second argument is y axis.
      fgScrollbar_ApplyPadding(self, msg->otherf, msg->otherfaux);
      return FG_ACCEPT;
    }
  }
  return fgControl_HoverMessage(&self->control, msg);
}