// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgMonitor.h"
#include "fgRoot.h"
#include "feathercpp.h"

void fgMonitor_Init(fgMonitor* BSS_RESTRICT self, fgFlag flags, fgRoot* BSS_RESTRICT parent, fgMonitor* BSS_RESTRICT prev, const AbsRect* coverage, const AbsVec* dpi)
{
  AbsVec scale = { (!dpi || !dpi->x || !parent->dpi.x) ? (FABS)1.0 : (parent->dpi.x / (FABS)dpi->x), (!dpi || !dpi->y || !parent->dpi.y) ? (FABS)1.0 : (parent->dpi.y / (FABS)dpi->y) };
  CRect& rootarea = parent->gui.element.transform.area;
  fgTransform transform = { {coverage->left*scale.x, 0, coverage->top*scale.y, 0, coverage->right*scale.x, 0, coverage->bottom*scale.y, 0,}, 0, {rootarea.left.abs, 0, rootarea.top.abs, 0} };
  self->coverage = *coverage;
  self->dpi = !dpi ? AbsVec_EMPTY : *dpi;
  self->mnext = !prev ? parent->monitors : 0;
  if(self->mnext) self->mnext->mprev = self;
  self->mprev = prev;
  if(self->mprev) self->mprev->mnext = self;
  else parent->monitors = self;

  fgElement_InternalSetup(&self->element, *parent, &prev->element, 0, flags, &transform, 0, (fgDestroy)&fgMonitor_Destroy, (fgMessage)&fgMonitor_Message);
}

void fgMonitor_Destroy(fgMonitor* self)
{
  assert(self != 0);
  if(self->mprev) self->mprev->mnext = self->mnext;
  if(self->mnext) self->mnext->mprev = self->mprev;
  self->mnext = self->mprev = 0;

  self->element.message = (fgMessage)&fgElement_Message;
  fgElement_Destroy(&self->element);
}

size_t fgMonitor_Message(fgMonitor* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CLONE:
    if(msg->e)
    {
      fgMonitor* hold = reinterpret_cast<fgMonitor*>(msg->e);
      memsubcpy<fgMonitor, fgElement>(hold, self);
      hold->mnext = 0;
      hold->mprev = 0;
      fgElement_Message(&self->element, msg);
    }
    return sizeof(fgMonitor);
  case FG_GETDPI:
    if(self->dpi.x != 0 && self->dpi.y != 0)
      return (size_t)&self->dpi;
    break;
  case FG_SETDPI:
  {
    self->dpi.x = msg->f;
    self->dpi.y = (size_t)msg->f2;
    AbsVec scale = { fgroot_instance->dpi.x / self->dpi.x, fgroot_instance->dpi.y / self->dpi.y };
    CRect area = { self->coverage.left*scale.x, 0, self->coverage.top*scale.y, 0, self->coverage.right*scale.x, 0, self->coverage.bottom*scale.y, 0 };
    size_t ret = self->element.SetArea(area);
    fgElement_Message(&self->element, msg); // Passes the SETDPI message to all children
    return ret;
  }
  case FG_DRAW: // Override draw call so we never clip, and replace the root DPI with our DPI
  {
    fgElement* hold = self->element.root;
    AbsRect curarea;
    fgDrawAuxData data = {
      sizeof(fgDrawAuxData),
      self->dpi,
      self->element.scaling,
      {0,0}
    };

    while(hold)
    {
      if(!(hold->flags&FGELEMENT_HIDDEN) && hold != fgroot_instance->topmost)
      {
        ResolveRectCache(hold, &curarea, (AbsRect*)msg->p, (hold->flags & FGELEMENT_BACKGROUND) ? 0 : &self->element.padding);
        _sendmsg<FG_DRAW, void*, void*>(hold, &curarea, &data);
      }
      hold = hold->next;
    }
    return FG_ACCEPT;
  }
  case FG_SETAREA:
  {
    AbsVec scale = { fgroot_instance->dpi.x / (float)self->dpi.x, fgroot_instance->dpi.y / (float)self->dpi.y };
    if(!msg->subtype)
    {
      size_t r = fgElement_Message(&self->element, msg);
      self->coverage = AbsRect { self->element.transform.area.left.abs / scale.x, self->element.transform.area.top.abs / scale.y, self->element.transform.area.right.abs / scale.x, self->element.transform.area.bottom.abs / scale.y };
      return r;
    }
    if(msg->p != 0)
    {
      self->coverage = *(AbsRect*)msg->p;
      CRect area = { self->coverage.left*scale.x, 0, self->coverage.top*scale.y, 0, self->coverage.right*scale.x, 0, self->coverage.bottom*scale.y, 0 };
      FG_Msg m = { 0 };
      m.type = FG_SETAREA;
      m.p = &area;
      return fgElement_Message(&self->element, &m);
    }
    fgLog(FGLOG_WARNING, "%s set NULL area", fgGetFullName(&self->element).c_str());
    return 0;
  }
  case FG_INJECT:
    return fgStandardInject(&self->element, (const FG_Msg*)msg->p, NULL);
  case FG_GETCLASSNAME:
    return (size_t)"Monitor";
  }

  return fgElement_Message(&self->element, msg);
}