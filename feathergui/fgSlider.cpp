// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSlider.h"
#include "bss-util\bss_util.h"

void FG_FASTCALL fgSlider_Init(fgSlider* BSS_RESTRICT self, size_t range, fgFlag flags, fgChild* BSS_RESTRICT parent, const fgElement* element)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, element, (FN_DESTROY)&fgSlider_Destroy, (FN_MESSAGE)&fgSlider_Message);
}
void FG_FASTCALL fgSlider_Destroy(fgSlider* self)
{
  fgWindow_Destroy(&self->window);
}
size_t FG_FASTCALL fgSlider_Message(fgSlider* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    fgChild_Init(&self->slider, FGCHILD_BACKGROUND, (fgChild*)self, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->slider);
    self->value = 0;
    self->range = 0;
    return 0;
  case FG_SETSTATE:
    if(msg->otherintaux)
    {
      if(self->range == msg->otherint)
        return 0;
      self->range = msg->otherint;
    }
    else
    {
      if(self->value == msg->otherint)
        return 0;
      self->value = msg->otherint;
    }
    CRect area = self->slider.element.area;
    area.left.rel = area.right.rel = !self->range ? 0.0 : ((FREL)self->value / (FREL)self->range);
    fgChild_VoidMessage(&self->slider, FG_SETAREA, &area);
    return 0;
  case FG_MOUSEMOVE:
    if(fgCaptureWindow == (fgChild*)self)
    {
      AbsRect out;
      ResolveRect((fgChild*)self, &out);
      double x = (out.left - msg->x)/(out.right - out.left); // we need all the precision in a double here
      size_t value = bss_util::fFastRound(bssclamp(x, 0.0, 1.0)*self->range); // Clamp to [0,1], multiply into [0, range], then round to nearest integer.
      fgChild_IntMessage(&self->slider, FG_SETSTATE, value, 0);
    }
    break;
  case FG_GETSTATE:
    return self->value;
  case FG_GETCLASSNAME:
    return (size_t)"fgSlider";
  }
  return fgWindow_Message(&self->window, msg);
}