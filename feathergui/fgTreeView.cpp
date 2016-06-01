// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTreeView.h"
#include "fgBox.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

size_t FG_FASTCALL fgTreeItemArrow_Message(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON && self->parent != 0)
      _sendmsg<FG_ACTION>(self->parent);
    break;
  }
  return fgElement_Message(self, msg);
}

void FG_FASTCALL fgTreeItem_Init(fgTreeItem* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(&self->element, parent, next, 0, flags, transform, (fgDestroy)&fgTreeItem_Destroy, (fgMessage)&fgTreeItem_Message);
}
size_t FG_FASTCALL fgTreeItem_Message(fgTreeItem* self, const FG_Msg* msg)
{
  static const size_t EXPANDED = ((size_t)1) << ((sizeof(size_t) << 3) - 1);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(&self->element, msg);
    fgElement_Init(&self->arrow, &self->element, 0, "fgTreeItem:arrow", FGELEMENT_BACKGROUND|FGELEMENT_HIDDEN, &fgTransform_EMPTY);
    self->arrow.message = (fgMessage)&fgTreeItemArrow_Message;
    self->count = 0;
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->element, 0, "hidden", fgStyleGetMask("visible", "hidden"));
    return FG_ACCEPT;
  case FG_LAYOUTFUNCTION:
    self->count += msg->subtype == FGELEMENT_LAYOUTADD;
    assert(msg->subtype != FGELEMENT_LAYOUTREMOVE || self->count > 0);
    self->count -= msg->subtype == FGELEMENT_LAYOUTREMOVE;
    self->element.SetFlag(FGELEMENT_HIDDEN, (self->count & (~EXPANDED)) != 0);
    return fgLayout_Tile(&self->element, (const FG_Msg*)msg->other, 1, (CRect*)msg->other2);
  case FG_ACTION:
    self->count = ((self->count&EXPANDED) ^ EXPANDED) | (self->count&(~EXPANDED));
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->element, 0, (self->count&EXPANDED) ? "visible" : "hidden", fgStyleGetMask("visible", "hidden"));
    return FG_ACCEPT;
  }

  return fgElement_Message(&self->element, msg);
}

void FG_FASTCALL fgTreeItem_Destroy(fgTreeItem* self)
{
  fgElement_Destroy(&self->element);
}
void FG_FASTCALL fgTreeView_Init(fgTreeView* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, 0, flags, transform, (fgDestroy)&fgTreeView_Destroy, (fgMessage)&fgTreeView_Message);
}
void FG_FASTCALL fgTreeView_Destroy(fgTreeView* self)
{
  fgScrollbar_Destroy(&self->scrollbar);
}
size_t FG_FASTCALL fgTreeView_Message(fgTreeView* self, const FG_Msg* msg)
{
  fgElement* hold = (fgElement*)msg->other;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgScrollbar_Message(&self->scrollbar, msg);
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(*self, 0, "visible", fgStyleGetMask("visible", "hidden"));
    return FG_ACCEPT;
  case FG_LAYOUTFUNCTION:
    return fgLayout_Tile(*self, (const FG_Msg*)msg->other, 1, (CRect*)msg->other2);
  case FG_GETCLASSNAME:
    return (size_t)"fgTreeView";
  }
  return fgScrollbar_Message(&self->scrollbar, msg);
}