// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgList.h"
#include "fgRoot.h"
#include "fgCurve.h"
#include "bss-util/bss_util.h"

static const char* FGSTR_LISTITEM = "ListItem";

void fgListItem_Init(fgControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgControl_Destroy, (fgMessage)&fgListItem_Message);
}

size_t fgListItem_Message(fgControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_DRAGOVER:
  case FG_DROP:
    return fgSendMessage(self->element.parent, msg);
  case FG_MOUSEMOVE:
  case FG_MOUSEDOWN:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
    fgSendMessage(self->element.parent, msg); // We send these messages to our parent FIRST, then override the resulting hover message by processing them ourselves.
    break;
  case FG_GETCLASSNAME:
    return (size_t)FGSTR_LISTITEM;
  }

  return fgControl_HoverMessage(self, msg);
}

void fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgList_Destroy, (fgMessage)&fgList_Message);
}
void fgList_Destroy(fgList* self)
{
  self->box->message = (fgMessage)fgBox_Message;
  fgBox_Destroy(&self->box);
}
void fgList_Draw(fgElement* self, const AbsRect* area, const fgDrawAuxData* data, fgElement* begin)
{
  fgList* realself = reinterpret_cast<fgList*>(self);
  for(size_t i = 0; i < realself->box.selected.l; ++i)
  {
    if(realself->box.selected.p[i]->GetClassName() != FGSTR_LISTITEM)
    {
      AbsRect r;
      ResolveRectCache(realself->box.selected.p[i], &r, area, (realself->box.selected.p[i]->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
      fgSnapAbsRect(r, self->flags);
      fgroot_instance->backend.fgDrawAsset(0, &CRect_EMPTY, realself->select.color, 0, 0.0f, &r, 0.0f, &AbsVec_EMPTY, FGRESOURCE_RECT, data);
    }
  }

  if(realself->mouse.state&FGMOUSE_DRAG)
  {
    AbsRect cache;
    fgElement* target = fgElement_GetChildUnderMouse(self, realself->mouse.x, realself->mouse.y, &cache);
    if(target)
    { 
      AbsRect r;
      ResolveRectCache(target, &r, (AbsRect*)&cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
      fgSnapAbsRect(r, self->flags);
      const AbsVec FULLVEC = { 1,1 };
      if(self->flags&FGBOX_TILEX)
      {
        float x = (realself->mouse.x > ((r.left + r.right) * 0.5f)) ? r.right : r.left;
        AbsVec line[2] = { { x, r.top }, { x, r.bottom - 1 } };
        fgroot_instance->backend.fgDrawLines(line, 2, realself->drag.color, &AbsVec_EMPTY, &FULLVEC, 0, &AbsVec_EMPTY, data);
      }
      else if(self->flags&FGBOX_TILEY)
      {
        float y = (realself->mouse.y > ((r.top + r.bottom) * 0.5f)) ? r.bottom : r.top;
        AbsVec line[2] = { { r.left, y }, { r.right - 1, y } };
        fgroot_instance->backend.fgDrawLines(line, 2, realself->drag.color, &AbsVec_EMPTY, &FULLVEC, 0, &AbsVec_EMPTY, data);
      }
    }
  }
  else
  {
    AbsRect cache;
    fgElement* target = fgElement_GetChildUnderMouse(self, realself->mouse.x, realself->mouse.y, &cache);
    if(target && target->GetClassName() != FGSTR_LISTITEM)
    {
      AbsRect r;
      ResolveRectCache(target, &r, &cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
      fgSnapAbsRect(r, self->flags);
      fgroot_instance->backend.fgDrawAsset(0, &CRect_EMPTY, realself->hover.color, 0, 0.0f, &r, 0.0f, &AbsVec_EMPTY, FGRESOURCE_RECT, data);
    }
  }

  fgBoxRenderDividers(self, realself->box.dividercolor, realself->box.dividerskin, area, area, data, begin);
}
fgElement* fgList_GetSplit(fgList* self, const FG_Msg* msg)
{
  if(self->splitter != 0 && (self->box->flags&FGBOX_TILE) && (self->box->flags&FGBOX_TILE) != FGBOX_TILE)
  {
    AbsRect cache;
    ResolveRect(*self, &cache);
    fgElement* cur = self->box->GetItemAt(msg->x, msg->y);
    if(cur != 0)
    {
      AbsRect child;
      ResolveRectCache(cur, &child, &cache, &self->box->padding);
      bool prev = (self->box->flags&FGBOX_TILEX) ? (msg->x < (child.left + child.right)*0.5f) : (msg->y < (child.top + child.bottom)*0.5f);
      fgElement* p = prev ? fgLayout_GetPrev(cur) : fgLayout_GetNext(cur);
      if(p || !prev) // You can't split if there's no previous element
      {
        AbsRect after = child; // By setting after to child we can resolve back to child instead of after if we're resolving a previous element
        if(p)
          ResolveRectCache(p, prev ? &child : &after, &cache, &self->box->padding);
        else
          after = AbsRect{ child.right, child.bottom, child.right, child.bottom };

        FABS mid = (self->box->flags&FGBOX_TILEX) ? (child.right + after.left)*0.5f : (child.bottom + after.top)*0.5f;
        FABS mpos = (self->box->flags&FGBOX_TILEX) ? msg->x : msg->y;
        if(abs(mpos - mid) <= self->splitter)
          return prev ? p : cur;
      }
    }
  }
  return 0;
}

size_t fgList_Message(fgList* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Message(&self->box, msg);
    bss::memsubset<fgList, fgBox>(self, 0);
    self->box.fndraw = &fgList_Draw;
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgList* hold = reinterpret_cast<fgList*>(msg->e);
      bss::memsubcpy<fgList, fgBox>(hold, self); // We do this first because we have to ensure our messages can process ADDCHILD correctly.
      memset(&hold->mouse, 0, sizeof(fgMouseState));
      hold->split = 0;
      fgBox_Message(&self->box, msg);
    }
    return sizeof(fgList);
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    if((self->split = fgList_GetSplit(self, msg)) != 0)
    {
      self->splitedge = (self->box->flags&FGBOX_TILEX) ? self->split->transform.area.left.abs + fgLayout_GetElementWidth(self->split) : self->split->transform.area.top.abs + fgLayout_GetElementHeight(self->split);
      self->splitmouse = (self->box->flags&FGBOX_TILEX) ? msg->x : msg->y;
      break;
    }
    if(self->box->flags&FGLIST_SELECT)
    {
      bool multi = (self->box->flags&FGLIST_MULTISELECT) == FGLIST_MULTISELECT;
      bool shift = fgroot_instance->GetKey(FG_KEY_SHIFT);

      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!target)
        break;
      size_t index = ((fgElementArray&)self->box.selected).Find(target);

      if(!multi || (!shift && (!(self->box->flags&FGLIST_DRAGGABLE) || index == (size_t)~0)))
      {
        fgBox_DeselectAll(&self->box);
        fgBox_SelectTarget(&self->box, target);
      }
      else if(shift)
      {
        if(index == (size_t)~0)
          fgBox_SelectTarget(&self->box, target);
        else
        {
          fgSetFlagStyle(self->box.selected.p[index], "selected", false);
          ((fgElementArray&)self->box.selected).Remove(index);
          self->box->Selection(0);
        }
      }
    }
    break;
  case FG_MOUSEUP:
    if((self->box->flags&FGLIST_DRAGGABLE) != 0 && (self->box->flags&FGLIST_MULTISELECT) == FGLIST_MULTISELECT && !fgroot_instance->GetKey(FG_KEY_SHIFT))
    {
      fgBox_DeselectAll(&self->box);
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!target)
      {
        self->box->Selection(0);
        break;
      }
      fgBox_SelectTarget(&self->box, target);
    }
    self->split = 0;
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEMOVE:
    fgUpdateMouseState(&self->mouse, msg);
    if(self->split) // check if we are actively dragging a splitter
    {
      CRect area = self->split->transform.area;
      if(self->box->flags&FGBOX_TILEX)
        area.right.abs = std::max(area.left.abs, self->splitedge + (msg->x - self->splitmouse));
      else
        area.bottom.abs = std::max(area.top.abs, self->splitedge + (msg->y - self->splitmouse));
      self->split->SetArea(area);
      return (self->box->flags&FGBOX_TILEX) ? FGCURSOR_RESIZEWE : FGCURSOR_RESIZENS;
    }
    if(fgList_GetSplit(self, msg) != 0)
      return (self->box->flags&FGBOX_TILEX) ? FGCURSOR_RESIZEWE : FGCURSOR_RESIZENS;
    if((self->box->flags&FGLIST_DRAGGABLE) && (self->mouse.state&FGMOUSE_INSIDE)) // Check if we clicked inside this window
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache); // find item below the mouse cursor (if any) and initiate a drag for it.
      if(target != 0)
      {
        fgroot_instance->backend.fgDragStart(FGCLIPBOARD_ELEMENT, target, target);
        fgroot_instance->fgCaptureWindow = 0;
        self->mouse.state &= ~FGMOUSE_INSIDE;
      }
    }
    break;
  case FG_MOUSEOFF:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_DRAGOVER:
    fgUpdateMouseState(&self->mouse, msg);
    if((fgroot_instance->dragtype == FGCLIPBOARD_ELEMENT) && (fgroot_instance->dragdata != 0) && (((fgElement*)fgroot_instance->dragdata)->parent == *self)) // Accept a drag element only if it's from this list
      return FGCURSOR_DRAG;
    break; // the default handler rejects it for us
  case FG_DROP:
    fgUpdateMouseState(&self->mouse, msg);
    if((fgroot_instance->dragtype == FGCLIPBOARD_ELEMENT) && (fgroot_instance->dragdata != 0))
    {
      fgElement* drag = (fgElement*)fgroot_instance->dragdata;
      if(drag->parent != *self)
        break; // drop to default handling to reject this if it isn't a child of this control

      AbsRect cache;
      AbsRect rect;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!target)
        break;
      ResolveRectCache(target, &rect, &cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &(*self)->padding);

      if(self->box->flags&FGBOX_TILEX)
      {
        if(self->mouse.x > ((rect.left + rect.right) * 0.5f)) // if true, it's after target, so move the target pointer up one.
          target = target->next;
      }
      else if(self->box->flags&FGBOX_TILEY)
      {
        if(self->mouse.y > ((rect.top + rect.bottom) * 0.5f)) // if true, it's after target, so move the target pointer up one.
          target = target->next;
      }

      // Remove the child from where it currently is, then re-insert it, but only if target is not drag
      if(target != drag)
      {
        drag->SetParent(0);
        self->box->AddChild(drag, target);
      }
      return FG_ACCEPT;
    }
    break;
  case FG_GETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_SELECT:
    case FGSETCOLOR_MAIN: return self->select.color;
    case FGSETCOLOR_HOVER: return self->hover.color;
    case FGSETCOLOR_DRAG: return self->drag.color;
    }
    break;
  case FG_SETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_SELECT:
    case FGSETCOLOR_MAIN: self->select.color = (unsigned int)msg->i; return FG_ACCEPT;
    case FGSETCOLOR_HOVER: self->hover.color = (unsigned int)msg->i; return FG_ACCEPT;
    case FGSETCOLOR_DRAG: self->drag.color = (unsigned int)msg->i; return FG_ACCEPT;
    }
    break;
  case FG_SETVALUE:
    if(!msg->subtype || msg->subtype == FGVALUE_FLOAT)
      self->splitter = msg->f;
    else if(msg->subtype == FGVALUE_INT64)
      self->splitter = (float)msg->i;
    else
    {
      fgLog(FGLOG_INFO, "%s set invalid value type: %hu", fgGetFullName(*self).c_str(), msg->subtype);
      return 0;
    }
    return FG_ACCEPT;
  case FG_GETVALUE:
    if(!msg->subtype || msg->subtype == FGVALUE_FLOAT)
      return *(size_t*)&self->splitter;
    if(msg->subtype == FGVALUE_INT64)
      return (size_t)self->splitter;
    fgLog(FGLOG_INFO, "%s requested invalid value type: %hu", fgGetFullName(*self).c_str(), msg->subtype);
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"List";
  }

  return fgBox_Message(&self->box, msg);
}