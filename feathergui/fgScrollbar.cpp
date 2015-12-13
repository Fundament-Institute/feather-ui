// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util\bss_util.h"

void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgWindow* parent, const fgElement* element, FG_UINT id, fgFlag flags)
{
  fgChild_InternalSetup((fgChild*)self, flags, parent, element, (FN_DESTROY)&fgSlider_Destroy, (FN_MESSAGE)&fgSlider_Message);
}
void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self)
{
  fgWindow_Destroy(&self->window);
}
void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{

}
size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgWindow_Message(&self->window, msg);
    fgChild_Init(&self->slider, FGCHILD_BACKGROUND, (fgChild*)self, &fgElement_CENTER);
    fgChild_AddPreChild((fgChild*)self, &self->slider);
    self->maxdim.x = 0;
    self->maxdim.y = 0;
    return 0;
  case FG_SETAREA:
  case FG_SETELEMENT:
  default: // By default, pass to our region
    return fgChild_PassMessage(&self->region, msg);
  }
  return fgWindow_Message(&self->window, msg);
}