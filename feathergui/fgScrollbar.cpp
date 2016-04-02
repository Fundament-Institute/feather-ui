// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgScrollbar_Destroy, (FN_MESSAGE)&fgScrollbar_Message);
}
void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self)
{
  fgWindow_Destroy(&self->window);
}

void FG_FASTCALL fgScrollbar_Redim(fgScrollbar* self, CRect& area)
{
  area.right.abs += self->barcache.x;
  area.bottom.abs += self->barcache.y;
  if(!(self->window.element.flags&FGCHILD_EXPANDX)) // If we actually are expanding along a given axis, use that result, otherwise replace it.
    area.right.abs = self->window.element.element.area.right.abs;
  if(!(self->window.element.flags&FGCHILD_EXPANDY))
    area.bottom.abs = self->window.element.element.area.bottom.abs;
  fgSendMsg<FG_SETAREA, void*>(*self, &area); // SETAREA will set MAXDIM appropriately
}

void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{
  AbsRect r;
  ResolveRect(*self, &r);
  AbsVec dim = { r.right - r.left, r.bottom - r.top };

  bool hideh = !!(self->window.element.flags&FGSCROLLBAR_HIDEH);
  bool hidev = !!(self->window.element.flags&FGSCROLLBAR_HIDEV);

  // We have to figure out which scrollbars are visible based on flags and our dimensions
  bool scrollx = !hideh && ((dim.x < self->realsize.x) || self->window.element.flags&FGSCROLLBAR_SHOWH);
  bool scrolly = !hidev && ((dim.y < self->realsize.y) || self->window.element.flags&FGSCROLLBAR_SHOWV);

  // Now things get tricky - we have to detect if enabling a scrollbar just caused us to shrink past the point where we need the other scrollbar
  scrollx = scrollx || (!hideh && scrolly && (dim.x < self->realsize.x + self->barcache.x));
  scrolly = scrolly || (!hidev && scrollx && (dim.y < self->realsize.y + self->barcache.y));

  // If we are appending a scrollbar that didn't exist before, this requires a padding change. This could change everything else (e.g. for textboxes)
  //if()
  {
    
  }
  
  // If we get this far, the padding has already been taken care of - append or remove scrollbars.


  // Set bar dimensions appropriately based on dimensions

  // Set bar positions appropriately based on padding
}

size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    memset(&self->maxdim, 0, sizeof(CVec));
    return 1;
  case FG_MOVE:
    if(!msg->other && (msg->otheraux & 6)) // detect an INTERNAL area change (every other area change will be handled elsewhere)
      fgScrollbar_Recalc(self); // If we have an actual area change, the scrollbars need to be repositioned or possibly removed entirely.
    break;
  case FG_SETAREA: // The set area only tries to enforce maxdim on the ABS portion. Consequently, responding to dimension changes must be done in FG_MOVE
    if(!msg->other)
      return 0;
    {
      AbsRect r;
      ResolveRect(*self, &r);
      AbsVec d = ResolveVec(&self->maxdim, &r);
      CRect* area = (CRect*)msg->other;
      if(self->maxdim.x.abs >= 0 && area->right.abs - area->left.abs > d.x)
        area->right.abs = area->left.abs + d.x;
      if(self->maxdim.y.abs >= 0 && area->bottom.abs - area->top.abs > d.y)
        area->bottom.abs = area->top.abs + d.y;

      char diff = CompareCRects(&self->window.element.element.area, area);
      memcpy(&self->window.element.element.area, area, sizeof(CRect));

      if(diff)
        fgChild_SubMessage(*self, FG_MOVE, FG_SETAREA, 0, diff);
    }
    return 1;
  case FG_SETPADDING:
    if(!msg->other)
      return 0;
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareMargins(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff) // Only send a move message if the change in padding could have resulted in an actual area change (this ensures a scroll delta change will NOT trigger an FG_MOVE)
        fgChild_SubMessage(*self, FG_MOVE, FG_SETPADDING, 0, diff | (1 << 8));
      else
        fgScrollbar_Recalc(self); // Recalculate scrollbar positions

    }
    return 1;
  case FG_LAYOUTCHANGE:
    {
      fgFlag flags = self->window.element.flags;
      auto area = self->window.element.element.area;
      self->window.element.flags |= FGCHILD_EXPAND;

      FG_Msg m = { 0 };
      m.type = FG_LAYOUTFUNCTION;
      m.other = (void*)msg;
      m.other2 = &area;
      size_t dim = fgChild_PassMessage(*self, &m);
      self->window.element.flags = flags;
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
      // clamp
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
      return fgChild_PassMessage(*self, &m);
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

      return fgChild_PassMessage(*self, &m);
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
  return fgWindow_HoverMessage(&self->window, msg);
}