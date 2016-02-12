// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util\bss_util.h"

void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgChild* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, element, (FN_DESTROY)&fgScrollbar_Destroy, (FN_MESSAGE)&fgScrollbar_Message);
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
  fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &area); // SETAREA will set MAXDIM appropriately
}

void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{
  // must take it account if the other scrollbar is visible

  // Append scrollbar - we will already have attempted to increase the area as necessary.
}
size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    memset(&self->maxdim, 0, sizeof(CVec));
    return 0;
  case FG_MOVE:
    if(!(msg->otheraux) && (msg->otheraux & 6)) // detect an area change
    {
      CRect area = self->window.element.element.area;
      area.right.abs = area.left.abs + self->realsize.x;
      area.bottom.abs = area.top.abs + self->realsize.y;
      fgScrollbar_Redim(self, area); // This appropriately removes the realsize if we aren't using the EXPAND flags, then calls SETAREA
      fgScrollbar_Recalc(self);
    }
    break;
  case FG_SETAREA: // The set area only tries to enforce maxdim on the ABS portion. Consequently, responding to dimension changes must be done in FG_MOVE
    if(!msg->other)
      return 1;
    {
      AbsRect r;
      ResolveRect((fgChild*)self, &r);
      AbsVec d = ResolveVec(&self->maxdim, &r);
      CRect* area = (CRect*)msg->other;
      if(self->maxdim.x.abs >= 0 && area->right.abs - area->left.abs > d.x)
        area->right.abs = area->left.abs + d.x;
      if(self->maxdim.y.abs >= 0 && area->bottom.abs - area->top.abs > d.y)
        area->bottom.abs = area->top.abs + d.y;

      char diff = CompareCRects(&self->window.element.element.area, area);
      memcpy(&self->window.element.element.area, area, sizeof(CRect));

      if(diff)
        fgChild_VoidAuxMessage((fgChild*)self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_SETPADDING:
    if(!msg->other)
      return 1;
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareAbsRects(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff)
        fgChild_VoidAuxMessage((fgChild*)self, FG_MOVE, 0, diff);
    }
    return 0;
  case FG_LAYOUTCHANGE:
    {
      fgFlag flags = self->window.element.flags;
      auto area = self->window.element.element.area;
      self->window.element.flags |= FGCHILD_EXPAND;

      FG_Msg m = { 0 };
      m.type = FG_LAYOUTFUNCTION;
      m.other1 = (void*)msg;
      m.other2 = &area;
      size_t dim = fgChild_PassMessage((fgChild*)self, &m);
      self->window.element.flags = flags;
      self->realsize = { area.right.abs - area.left.abs, area.bottom.abs - area.top.abs }; // retrieve real area and then reset to the area of the window.
      if(dim)
        fgScrollbar_Redim(self, area);
      return 0;
    }
    break;
  case FG_ACTION:
    switch(msg->subtype)
    {
    case FGSCROLLBAR_CHANGE: // corresponds to an actual change amount. First argument is x axis, second argument is y axis.
      // apply change to padding
      // clamp
      // derive new scrollbar positions
      return 0;
    case FGSCROLLBAR_PAGE: // By default a page scroll (clicking on the area outside of the bar or hitting pageup/pagedown) attempts to scroll by the width of the container in that direction.
    {
      AbsRect r;
      ResolveRect((fgChild*)self, &r);
      FG_Msg m { 0 };
      m.type = FG_ACTION;
      m.subtype = FGSCROLLBAR_CHANGE;
      if(msg->otherint & 0) m.otherf = (r.right - r.left)*(msg->otherint == 0 ? -1 : 1);
      if(msg->otherint & 1) m.otherfaux = (r.bottom - r.top)*(msg->otherint == 1 ? -1 : 1);
      return fgChild_PassMessage((fgChild*)self, &m);
    }
    case FGSCROLLBAR_HDELTA:
    case FGSCROLLBAR_VDELTA:
    {
      FG_Msg m { 0 };
      m.type = FG_ACTION;
      m.subtype = FGSCROLLBAR_CHANGE;
      if(msg->subtype == FGSCROLLBAR_HDELTA)
        m.otherfaux = msg->otherint;
      else
        m.otherf = msg->otherint;

      return fgChild_PassMessage((fgChild*)self, &m);
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
        ResolveRect((fgChild*)&self->btn[4], &r);
        m.otherf = (r.left - r.right); // negate size
        break;
      case 1:
        ResolveRect((fgChild*)&self->btn[0], &r);
        m.otherfaux = (r.top - r.bottom); // negate size
        break;
      case 2:
        ResolveRect((fgChild*)&self->btn[3], &r);
        m.otherf = (r.right - r.left);
        break;
      case 3:
        ResolveRect((fgChild*)&self->btn[1], &r);
        m.otherfaux = (r.bottom - r.top);
        break;
      }
    }
      break;
    }
  }
  return fgWindow_Message(&self->window, msg);
}