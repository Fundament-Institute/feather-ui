// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTabControl.h"
#include "feathercpp.h"

void FG_FASTCALL fgTabControl_Init(fgTabControl* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgTabControl_Destroy, (fgMessage)&fgTabControl_Message);
}

void FG_FASTCALL fgTabControl_Destroy(fgTabControl* self)
{
  fgBox_Destroy(&self->header);
  fgControl_Destroy(&self->control);
}

size_t FG_FASTCALL fgTabControl_Message(fgTabControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_HoverMessage(&self->control, msg);
    fgBox_Init(&self->header, *self, 0, "fgTabControl$header", 0, &fgTransform_EMPTY);
    return FG_ACCEPT;
  case FG_ADDITEM:
  {
    fgElement* button = fgCreate("RadioButton", self->header, 0, "fgTabControl$toggle", FGELEMENT_EXPANDX, &fgTransform_EMPTY);
    fgElement* panel = fgCreate("fgElement", *self, 0, "fgTabControl$panel", 0, &fgTransform_DEFAULT);
    panel->userdata = button;
    button->userdata = panel;
    button->SetText((const char*)msg->other);
    return FG_ACCEPT;
  }
  }

  return fgControl_HoverMessage(&self->control, msg);
}

fgElement* fgTabControl::AddItem(const char* name) { return (fgElement*)_sendmsg<FG_ADDITEM, const void*>(*this, name); }
