// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSlider.h"
#include "bss-util/bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgSlider_Init(fgSlider* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgSlider_Destroy, (fgMessage)&fgSlider_Message);
}
void FG_FASTCALL fgSlider_Destroy(fgSlider* self)
{
  fgControl_Destroy(&self->control);
}
size_t FG_FASTCALL fgSlider_Message(fgSlider* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    fgElement_Init(&self->slider, *self, 0, "Slider$slider", FGELEMENT_EXPAND | FGELEMENT_IGNORE, &fgTransform_CENTER, 0);
    self->value = 0;
    self->range = 0;
    return FG_ACCEPT;
  case FG_SETVALUE:
    if(msg->subtype <= FGVALUE_FLOAT)
    {
      ptrdiff_t value = (msg->subtype == FGVALUE_FLOAT) ? bss_util::fFastRound(msg->otherf) : msg->otherint;
      if(msg->otheraux)
      {
        if(self->range == msg->otherint)
          return FG_ACCEPT;
        self->range = value;
      }
      else
      {
        if(self->value == msg->otherint)
          return FG_ACCEPT;
        self->value = value;
      }
      CRect area = self->slider.transform.area;
      area.left.rel = area.right.rel = !self->range ? 0.0f : ((FREL)self->value / (FREL)self->range);
      _sendmsg<FG_SETAREA, void*>(&self->slider, &area);
      return FG_ACCEPT;
    }
    return 0;
  case FG_MOUSEMOVE:
  case FG_MOUSEUP:
    if(fgCaptureWindow != *self)
      break;
  case FG_MOUSEDOWN:
  {
    AbsRect out;
    ResolveRect(*self, &out);
    out.left += self->control.element.padding.left;
    out.top += self->control.element.padding.top;
    out.right -= self->control.element.padding.right;
    out.bottom -= self->control.element.padding.bottom;

    double x = (msg->x - out.left) / (out.right - out.left); // we need all the precision in a double here
    size_t value = bss_util::fFastRound(bssclamp(x, 0.0, 1.0)*self->range); // Clamp to [0,1], multiply into [0, range], then round to nearest integer.
    fgIntMessage(*self, FG_SETVALUE, value, 0);
  }
    break;
  case FG_GETVALUE:
    if(!msg->subtype || msg->subtype == FGVALUE_INT64)
      return msg->otherint ? self->range : self->value;
    if(msg->subtype == FGVALUE_FLOAT)
    {
      float val = (float)(msg->otherint ? self->range : self->value);
      return *(size_t*)&val;
    }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"Slider";
  }
  return fgControl_HoverMessage(&self->control, msg);
}