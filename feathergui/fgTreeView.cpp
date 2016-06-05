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

void FG_FASTCALL fgTreeItem_Init(fgTreeItem* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(&self->control.element, parent, next, name, flags, transform, (fgDestroy)&fgTreeItem_Destroy, (fgMessage)&fgTreeItem_Message);
}
size_t FG_FASTCALL fgTreeItem_Message(fgTreeItem* self, const FG_Msg* msg)
{
  static const char* CLASSNAME = "fgTreeItem";
  static const char* ARROWNAME = "fgTreeItem:arrow";
  static const size_t EXPANDED = ((size_t)1) << ((sizeof(size_t) << 3) - 1);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    fgElement_Init(&self->arrow, &self->control.element, 0, ARROWNAME, FGELEMENT_BACKGROUND|FGELEMENT_HIDDEN, &fgTransform_EMPTY);
    self->arrow.message = (fgMessage)&fgTreeItemArrow_Message;
    self->count = EXPANDED;
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->arrow, 0, "visible", fgStyleGetMask("visible", "hidden"));
    return FG_ACCEPT;
  case FG_ADDITEM:
  case FG_ADDCHILD:
    if(((fgElement*)msg->other)->GetClassName() == CLASSNAME)
      self->arrow.SetFlag(FGELEMENT_HIDDEN, ((++self->count) & (~EXPANDED)) == 0);
    break;
  case FG_REMOVEITEM:
  case FG_REMOVECHILD:
    if(((fgElement*)msg->other)->GetClassName() == CLASSNAME)
    {
      assert(self->count > 0);
      self->arrow.SetFlag(FGELEMENT_HIDDEN, ((--self->count) & (~EXPANDED)) == 0);
    }
    break;
  case FG_LAYOUTFUNCTION:
    return fgLayout_Tile(&self->control.element, (const FG_Msg*)msg->other, FGBOX_TILEY, (CRect*)msg->other2);
  case FG_ACTION:
    if(msg->subtype != 0) // If nonzero this action was meant for the root
      return !self->control.element.parent ? 0 : fgPassMessage(self->control.element.parent, msg);

    self->count = ((self->count&EXPANDED) ^ EXPANDED) | (self->count&(~EXPANDED));
    _sendsubmsg<FG_SETSTYLE, void*, size_t>(&self->arrow, 0, (self->count&EXPANDED) ? "visible" : "hidden", fgStyleGetMask("visible", "hidden"));
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
      _sendsubmsg<FG_ACTION, void*>(self->control.element.parent, FGSCROLLBAR_SCROLLTO, &r);
    }
    break;
  }

  return fgControl_Message(&self->control, msg);
}

void FG_FASTCALL fgTreeItem_Destroy(fgTreeItem* self)
{
  fgControl_Destroy(&self->control);
}
void FG_FASTCALL fgTreeView_Init(fgTreeView* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
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
    return FG_ACCEPT;
  case FG_LAYOUTFUNCTION:
    return fgLayout_Tile(*self, (const FG_Msg*)msg->other, FGBOX_TILEY, (CRect*)msg->other2);
  case FG_GETCLASSNAME:
    return (size_t)"fgTreeView";
  case FG_GOTFOCUS:
    if(fgElement_CheckLastFocus(*self)) // try to resolve via lastfocus
      return FG_ACCEPT;
    break;
  }
  return fgScrollbar_Message(&self->scrollbar, msg);
}