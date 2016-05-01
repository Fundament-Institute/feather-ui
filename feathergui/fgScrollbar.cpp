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
  if(self->barcache.y >= 0)
  {
    float dim = area.bottom - area.top - self->realpadding.top - self->realpadding.bottom;
    float length = self->realsize.y / dim;
    float pos = self->control.element.padding.top / (self->realsize.y - dim);
    self->btn[2]->SetArea(CRect { 0, 0, 0, (1.0f - length)*pos, 0, 1.0f, 0, (1.0f - length)*pos + length });
  }
  if(self->barcache.x >= 0)
  {
    float dim = area.right - area.left - self->realpadding.left - self->realpadding.right;
    float length = self->realsize.x / dim;
    float pos = self->control.element.padding.left / (self->realsize.x - dim);
    self->btn[5]->SetArea(CRect { 0, (1.0f - length)*pos, 0, 0, 0, (1.0f - length)*pos + length, 0, 1.0f });
  }
}
void FG_FASTCALL fgScrollbar_SetPadding(fgScrollbar* self)
{

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
  if(scrollx ^ (self->barcache.x < 0) || scrolly ^ (self->barcache.y < 0))
  {
    if(scrollx ^ (self->barcache.x < 0)) self->barcache.x = -self->barcache.x;
    if(scrolly ^ (self->barcache.y < 0)) self->barcache.y = -self->barcache.y;
    fgScrollbar_SetPadding(self);
    return fgScrollbar_Recalc(self);
  }
  
  // If we get this far, the padding has already been taken care of - append or remove scrollbars.
  self->bg[1].SetFlag(FGELEMENT_HIDDEN | FGELEMENT_IGNORE, scrollx);
  self->bg[0].SetFlag(FGELEMENT_HIDDEN | FGELEMENT_IGNORE, scrolly);

  // Set bar dimensions appropriately based on padding
  fgScrollbar_SetBars(self);
}

size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    memset(&self->maxdim, 0, sizeof(CVec));
    return 1;
  case FG_MOVE:
    if(!msg->other && (msg->otheraux & 6)) // detect an INTERNAL area change (every other area change will be handled elsewhere)
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
        fgSubMessage(*self, FG_MOVE, FG_SETPADDING, 0, diff | (1 << 8));
      else
        fgScrollbar_Recalc(self); // Recalculate scrollbar positions
      return 1;
    }
    return 0;
  case FG_LAYOUTCHANGE:
    {
      fgFlag flags = self->control.element.flags;
      auto area = self->control.element.transform.area;
      self->control.element.flags |= FGELEMENT_EXPAND;
      
      size_t dim = _sendmsg<FG_LAYOUTFUNCTION, const void*, void*>(*self, msg, &area);
      self->control.element.flags = flags;
      self->realsize = { area.right.abs - area.left.abs, area.bottom.abs - area.top.abs }; // retrieve real area and then reset to the area of the window.
      if(dim)
        fgScrollbar_Redim(self, area);
      return 1;
    }
    break;
  case FG_ACTION:
    switch(msg->subtype)
    {
    case FGSCROLLBAR_CHANGE: // corresponds to an actual change amount. First argument is x axis, second argument is y axis.
      // apply change to padding
      // clamp to [0,1] range so we don't overscroll outside of the control
      // Set new padding (which will recalculate scrollbar positions)
      return 1;
    case FGSCROLLBAR_PAGE: // By default a page scroll (clicking on the area outside of the bar or hitting pageup/pagedown) attempts to scroll by the width of the container in that direction.
    {
      AbsRect r;
      ResolveRect(*self, &r);
      FG_Msg m { 0 };
      m.type = FG_ACTION;
      m.subtype = FGSCROLLBAR_CHANGE;
      if(msg->otherint & 0) m.otherf = (r.right - r.left)*(msg->otherint == 0 ? -1 : 1);
      if(msg->otherint & 1) m.otherfaux = (r.bottom - r.top)*(msg->otherint == 1 ? -1 : 1);
      return fgPassMessage(*self, &m);
    }
    case FGSCROLLBAR_DELTAH:
    case FGSCROLLBAR_DELTAV:
    {
      FG_Msg m { 0 };
      m.type = FG_ACTION;
      m.subtype = FGSCROLLBAR_CHANGE;
      if(msg->subtype == FGSCROLLBAR_DELTAH)
        m.otherfaux = msg->otherint;
      else
        m.otherf = msg->otherint;

      return fgPassMessage(*self, &m);
    }
    case FGSCROLLBAR_BUTTON: // this should almost always be the lineheight, but we don't have that information so we just try the size of the button instead.
    {
      AbsRect r;
      FG_Msg m { 0 };
      m.type = FG_ACTION;
      m.subtype = FGSCROLLBAR_CHANGE;
      switch(msg->otherint)
      {
      case 0:
        ResolveRect(self->btn[4], &r);
        m.otherf = (r.left - r.right); // negate size
        break;
      case 1:
        ResolveRect(self->btn[0], &r);
        m.otherfaux = (r.top - r.bottom); // negate size
        break;
      case 2:
        ResolveRect(self->btn[3], &r);
        m.otherf = (r.right - r.left);
        break;
      case 3:
        ResolveRect(self->btn[1], &r);
        m.otherfaux = (r.bottom - r.top);
        break;
      }
    }
      break;
    }
  }
  return fgControl_HoverMessage(&self->control, msg);
}