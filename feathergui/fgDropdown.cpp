// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgDropdown.h"
#include "feathercpp.h"

void FG_FASTCALL fgDropdown_Init(fgControl* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, (fgDestroy)&fgBox_Destroy, (fgMessage)&fgDropdown_Message);
}

size_t FG_FASTCALL fgDropdown_Message(fgControl* self, const FG_Msg* msg)
{

  fgBox_Message(&self->box, msg);
}