// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgSlider.h"
#include "bss-util/bss_util.h"
#include "feathercpp.h"

void fgSlider_Init(fgSlider* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgSlider_Destroy, (fgMessage)&fgSlider_Message);
}
void fgSlider_Destroy(fgSlider* self)
{
  self->control->message = (fgMessage)fgControl_Message;
  fgControl_Destroy(&self->control);
}
size_t fgSlider_Message(fgSlider* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    fgElement_Init(&self->slider, *self, 0, "Slider$slider", FGELEMENT_EXPAND | FGELEMENT_IGNORE | FGFLAGS_INTERNAL, &fgTransform_CENTER, 0);
    self->value = 0;
    self->range = 0;
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgSlider* hold = reinterpret_cast<fgSlider*>(msg->e);
      fgControl_Message(&self->control, msg);
      self->slider.Clone(&hold->slider);
      hold->range = self->range;
      hold->value = self->value;
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, &hold->slider);
    }
    return sizeof(fgSlider);
  case FG_SETRANGE:
  case FG_SETVALUE:
    if(msg->subtype <= FGVALUE_FLOAT)
    {
      ptrdiff_t value = (msg->subtype == FGVALUE_FLOAT) ? bss::fFastRound(msg->f) : msg->i;
      if(msg->type == FG_SETRANGE)
      {
        if(self->range == value)
          return FG_ACCEPT;
        self->range = value;
      }
      else
      {
        if(self->value == value)
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
    if(fgroot_instance->fgCaptureWindow != *self)
      break;
  case FG_MOUSEDOWN:
  {
    AbsRect inner;
    ResolveInnerRect(*self, &inner);

    double x = (msg->x - inner.left) / (inner.right - inner.left); // we need all the precision in a double here
    size_t value = bss::fFastRound(bssclamp(x, 0.0, 1.0)*self->range); // Clamp to [0,1], multiply into [0, range], then round to nearest integer.
    _sendmsg<FG_SETVALUE, size_t>(*self, value);
  }
    break;
  case FG_GETRANGE:
    if(!msg->subtype || msg->subtype == FGVALUE_INT64)
      return self->range;
    if(msg->subtype == FGVALUE_FLOAT)
    {
      float range = self->range;
      return *(size_t*)&range;
    }
    return 0;
  case FG_GETVALUE:
    if(!msg->subtype || msg->subtype == FGVALUE_INT64)
      return self->value;
    if(msg->subtype == FGVALUE_FLOAT)
    {
      float value = self->value;
      return *(size_t*)&value;
    }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"Slider";
  }
  return fgControl_HoverMessage(&self->control, msg);
}