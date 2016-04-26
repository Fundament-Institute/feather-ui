// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgMonitor.h"
#include "fgRoot.h"
#include "feathercpp.h"

void FG_FASTCALL fgMonitor_Init(fgMonitor* BSS_RESTRICT self, fgFlag flags, fgRoot* BSS_RESTRICT parent, fgMonitor* BSS_RESTRICT prev, const AbsRect* coverage, size_t dpi)
{
  float scale = (!dpi || !parent->dpi) ? 1.0 : (parent->dpi / (float)dpi);
  fgElement element = { coverage->left*scale, 0, coverage->top*scale, 0, coverage->right*scale, 0, coverage->bottom*scale, 0, 0, 0, 0 };
  fgChild_InternalSetup(&self->element, flags, *parent, &prev->element, &element, (FN_DESTROY)&fgMonitor_Destroy, (FN_MESSAGE)&fgMonitor_Message);
  self->coverage = *coverage;
  self->dpi = dpi;
  self->mnext = !prev ? parent->monitors : 0;
  if(self->mnext) self->mnext->mprev = self;
  self->mprev = prev;
  if(self->mprev) self->mprev->mnext = self;
}

void FG_FASTCALL fgMonitor_Destroy(fgMonitor* self)
{
  assert(self != 0);
  if(self->mprev) self->mprev->mnext = self->mnext;
  if(self->mnext) self->mnext->mprev = self->mprev;
  self->mnext = self->mprev = 0;

  fgChild_Destroy(&self->element);
}

size_t FG_FASTCALL fgMonitor_Message(fgMonitor* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_GETDPI:
    return self->dpi;
  case FG_SETDPI:
  {
    self->dpi = (size_t)msg->otherint;
    float scale = fgroot_instance->dpi / (float)self->dpi;
    CRect area = { self->coverage.left*scale, 0, self->coverage.top*scale, 0, self->coverage.right*scale, 0, self->coverage.bottom*scale, 0 };
    size_t ret = self->element.SetArea(area);
    fgChild_Message(&self->element, msg); // Passes the SETDPI message to all children
    return ret;
  }
  case FG_DRAW: // Override draw call so we never clip, and replace the root DPI with our DPI
  {
    fgChild* hold = self->element.last; // we draw backwards through our list.
    AbsRect curarea;

    while(hold)
    {
      if(!(hold->flags&FGCHILD_HIDDEN))
      {
        ResolveRectCache(hold, &curarea, (AbsRect*)msg->other, (hold->flags & FGCHILD_BACKGROUND) ? 0 : &self->element.padding);
        _sendmsg<FG_DRAW, void*, size_t>(hold, &curarea, self->dpi);
      }
      hold = hold->prev;
    }
  }
  }

  return fgChild_Message(&self->element, msg);
}