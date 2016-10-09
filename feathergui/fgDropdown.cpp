// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgDropdown.h"
#include "feathercpp.h"

void FG_FASTCALL fgDropdown_Init(fgDropdown* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgBox_Destroy, (fgMessage)&fgDropdown_Message);
}

size_t FG_FASTCALL fgDropdown_Message(fgDropdown* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    self->selected = 0;
    self->hover.color = 0x99999999;
    break;
  case FG_MOUSEDOWN:
    if(fgCaptureWindow == *self)
      return fgControl_Message(&self->box.window.control, msg);
    break;
  case FG_MOUSEUP:
    if(fgCaptureWindow != *self)
      return fgControl_Message(&self->box.window.control, msg);
    break;
  case FG_MOUSEMOVE:
    if(fgCaptureWindow == *self)
      break;
    return FG_ACCEPT; // If the dropdown hasn't been dropped down, eat the mousemove events.
  case FG_DRAW:
    if(fgCaptureWindow != *self)
    { // In this case, we only render background elements and the single selected item (if there is one). Because of this, there's no need to use the ordered rendering because by definition we aren't rendering anything that would be ordered.
      char culled = (msg->subtype & 1);
      fgElement* hold = culled ? (*self)->rootnoclip : (*self)->root;
      AbsRect curarea;
      bool clipping = false;

      while(hold)
      {
        if((hold->flags & FGELEMENT_BACKGROUND) || hold == self->selected)
          fgStandardDrawElement(*self, hold, (AbsRect*)msg->other, msg->otheraux, curarea, clipping);
        hold = culled ? hold->nextnoclip : hold->next;
      }

      if(clipping)
        fgPopClipRect();

      return FG_ACCEPT;
    }
    break; // otherwise we render the whole thing normally
  case FG_GETCOLOR:
    return self->hover.color;
  case FG_SETCOLOR:
    self->hover.color = (size_t)msg->otherint;
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return (size_t)self->selected;
  case FG_GETCLASSNAME:
    return (size_t)"Dropdown";
  }
  fgBox_Message(&self->box, msg);
}