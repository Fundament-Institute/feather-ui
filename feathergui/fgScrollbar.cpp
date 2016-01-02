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
void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{
  // must take it account if the other scrollbar is visible
}
size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    fgChild_Init(&self->region, FGCHILD_IGNORE | FGCHILD_EXPAND, 0, &fgElement_CENTER);
    fgChild_VoidAuxMessage(&self->region, FG_SETPARENT, (fgChild*)self, 1);
    fgChild_AddPreChild((fgChild*)self, &self->region);
    memset(&self->maxdim, 0, sizeof(CVec));
    return 0;
  case FG_SETPARENT:
    if(msg->otheraux) // if nonzero, break to our actual handler, otherwise pass to the region
      break;
  case FG_ADDCHILD:
  case FG_ADDITEM:
    return fgChild_PassMessage(&self->region, msg);
  case FG_MOVE:
    if(!(msg->otheraux) && (msg->otheraux & 6))
    {
      AbsRect r;
      ResolveRect((fgChild*)self, &r);
      AbsVec d = ResolveVec(&self->maxdim, &r);
      AbsRect mod = r;
      if((self->window.element.flags&FGCHILD_EXPANDX) && (r.right - r.left) > d.x)
        mod.right = r.left + d.x;
      if((self->window.element.flags&FGCHILD_EXPANDY) && (r.bottom - r.top) > d.y)
        mod.bottom = r.top + d.y;

      if(mod.right != r.right || mod.bottom != r.bottom)
      {
        CRect area = self->window.element.element.area;
        area.right.abs += mod.right - r.right;
        area.bottom.abs += mod.bottom - r.bottom;
        fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &area); // this shouldn't trigger itself again because the mods should cancel out and it wll exit without doing anything.
        fgScrollbar_Recalc(self);
      }
    }
    break;
  }
  return fgWindow_Message(&self->window, msg);
}