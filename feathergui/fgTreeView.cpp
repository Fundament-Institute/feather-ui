// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTreeview.h"
#include "fgBox.h"
#include "bss-util/bss_util.h"
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

void FG_FASTCALL fgTreeItem_Init(fgTreeItem* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(&self->control.element, parent, next, name, flags, transform, units, (fgDestroy)&fgTreeItem_Destroy, (fgMessage)&fgTreeItem_Message);
}
size_t FG_FASTCALL fgTreeItem_Message(fgTreeItem* self, const FG_Msg* msg)
{
  static const char* CLASSNAME = "TreeItem";
  static const char* ARROWNAME = "TreeItem$arrow";
  static const size_t EXPANDED = ((size_t)1) << ((sizeof(size_t) << 3) - 1);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    fgElement_Init(&self->arrow, &self->control.element, 0, ARROWNAME, FGELEMENT_BACKGROUND|FGELEMENT_HIDDEN, 0, 0);
    self->arrow.message = (fgMessage)&fgTreeItemArrow_Message;
    self->count = EXPANDED;
    fgMaskSetStyle(&self->arrow, "visible", fgStyleGetMask("visible", "hidden"));
    return FG_ACCEPT;
  case FG_ADDITEM:
    if(msg->subtype != 0)
      return 0;
  case FG_ADDCHILD:
    if(((fgElement*)msg->other)->GetClassName() == CLASSNAME)
      self->arrow.SetFlag(FGELEMENT_HIDDEN, ((++self->count) & (~EXPANDED)) == 0);
    break;
  case FG_REMOVECHILD:
    if(((fgElement*)msg->other)->GetClassName() == CLASSNAME)
    {
      assert(self->count > 0);
      self->arrow.SetFlag(FGELEMENT_HIDDEN, ((--self->count) & (~EXPANDED)) == 0);
    }
    break;
  case FG_LAYOUTFUNCTION:
    return fgTileLayout(&self->control.element, (const FG_Msg*)msg->other, FGBOX_TILEY, (AbsVec*)msg->other2);
  case FG_ACTION:
    if(msg->subtype != 0) // If nonzero this action was meant for the root
      return !self->control.element.parent ? 0 : fgPassMessage(self->control.element.parent, msg);

    self->count = ((self->count&EXPANDED) ^ EXPANDED) | (self->count&(~EXPANDED));
    fgMaskSetStyle(&self->arrow, (self->count&EXPANDED) ? "visible" : "hidden", fgStyleGetMask("visible", "hidden"));
    {
      fgElement* cur = self->control.element.root;
      while(cur)
      {
        if(cur->GetClassName() == CLASSNAME)
          cur->SetFlag(FGELEMENT_IGNORE | FGELEMENT_HIDDEN | FGELEMENT_BACKGROUND, !(self->count&EXPANDED));
        cur = cur->next;
      }
    }
    return FG_ACCEPT;
  case FG_GETCLASSNAME:
    return (size_t)CLASSNAME;
  case FG_GOTFOCUS:
    if(self->control.element.parent != 0)
    {
      AbsRect r;
      ResolveRect(&self->control.element, &r);
      _sendsubmsg<FG_ACTION, void*>(self->control.element.parent, FGSCROLLBAR_SCROLLTOABS, &r);
    }
    break;
  }

  return fgControl_Message(&self->control, msg);
}

void FG_FASTCALL fgTreeItem_Destroy(fgTreeItem* self)
{
  fgControl_Destroy(&self->control);
}
void FG_FASTCALL fgTreeview_Init(fgTreeview* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, 0, flags, transform, units, (fgDestroy)&fgTreeview_Destroy, (fgMessage)&fgTreeview_Message);
}
void FG_FASTCALL fgTreeview_Destroy(fgTreeview* self)
{
  fgScrollbar_Destroy(&self->scrollbar);
}
size_t FG_FASTCALL fgTreeview_Message(fgTreeview* self, const FG_Msg* msg)
{
  fgElement* hold = (fgElement*)msg->other;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgScrollbar_Message(&self->scrollbar, msg);
    return FG_ACCEPT;
  case FG_LAYOUTFUNCTION:
    return fgTileLayout(*self, (const FG_Msg*)msg->other, FGBOX_TILEY, (AbsVec*)msg->other2);
  case FG_GETCLASSNAME:
    return (size_t)"TreeView";
  case FG_GOTFOCUS:
    if(fgElement_CheckLastFocus(*self)) // try to resolve via lastfocus
      return FG_ACCEPT;
    break;
  }
  return fgScrollbar_Message(&self->scrollbar, msg);
}