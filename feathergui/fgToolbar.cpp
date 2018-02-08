// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgToolbar.h"

void fgToolbar_Init(fgToolbar* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgToolbar_Destroy, (fgMessage)&fgToolbar_Message);
}
void fgToolbar_Destroy(fgToolbar* self)
{
  self->box->message = (fgMessage)fgBox_Message;
  fgBox_Destroy(&self->box);
}
size_t fgToolbar_Message(fgToolbar* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ADDITEM:
    return (size_t)fgCreate("ToolGroup", *self, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
  case FG_GETCLASSNAME:
    return (size_t)"Toolbar";
  }
  return fgBox_Message(&self->box, msg);
}

void fgToolGroup_Init(fgBox* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgBox_Destroy, (fgMessage)&fgToolGroup_Message);
}
size_t fgToolGroup_Message(fgBox* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_ADDITEM:
    return (size_t)fgCreate("Box", *self, 0, 0, FGBOX_TILEX | FGELEMENT_EXPAND, 0, 0);
  case FG_GETCLASSNAME:
    return (size_t)"ToolGroup";
  }

  return fgBox_Message(self, msg);
}