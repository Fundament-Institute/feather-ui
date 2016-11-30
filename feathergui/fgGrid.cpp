// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgGrid.h"
#include "bss-util/cDynArray.h"

void FG_FASTCALL fgGrid_Init(fgGrid* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgGrid_Destroy, (fgMessage)&fgGrid_Message);
}
void FG_FASTCALL fgGrid_Destroy(fgGrid* self)
{
  fgBox_Destroy(&self->box);
}
size_t FG_FASTCALL fgGrid_Message(fgGrid* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Message(&self->box, msg);
    fgList_Init(&self->header, *self, 0, "Grid$header", FGELEMENT_BACKGROUND | FGELEMENT_EXPAND | FGBOX_TILEX, &fgTransform_EMPTY, 0);
    memset(&self->selected, 0, sizeof(fgVectorElement));
    self->hover.color = 0x99999999;
    break;
  case FG_ADDITEM:


    break;
  }

  return fgBox_Message(&self->box, msg);
}

void FG_FASTCALL fgGridRow_Init(fgGridRow* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgGridRow_Destroy, (fgMessage)&fgGridRow_Message);

}
void FG_FASTCALL fgGridRow_Destroy(fgGridRow* self)
{
  fgBox_Destroy(&self->box);
}

size_t FG_FASTCALL fgGridRow_Message(fgGridRow* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ADDITEM:
    break;
  }
  return fgBox_Message(&self->box, msg);
}