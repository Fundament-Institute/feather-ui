// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgList.h"
#include "fgRoot.h"
#include "fgCurve.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

static const char* FGSTR_LISTITEM = "ListItem";

void FG_FASTCALL fgListItem_Init(fgControl* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgElement_Destroy, (fgMessage)&fgListItem_Message);
}

size_t FG_FASTCALL fgListItem_Message(fgControl* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_DRAGOVER:
  case FG_DROP:
    return fgPassMessage(self->element.parent, msg);
  case FG_MOUSEMOVE:
  case FG_MOUSEDOWN:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEUP:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
    fgPassMessage(self->element.parent, msg); // We send these messages to our parent FIRST, then override the resulting hover message by processing them ourselves.
    break;
  case FG_NUETRAL:
    fgStandardNuetralSetStyle(*self, "nuetral");
    return FG_ACCEPT;
  case FG_HOVER:
    fgStandardNuetralSetStyle(*self, "hover");
    return FG_ACCEPT;
  case FG_ACTIVE:
    fgStandardNuetralSetStyle(*self, "active");
    return FG_ACCEPT;
  case FG_GETCLASSNAME:
    return (size_t)FGSTR_LISTITEM;
  }

  return fgControl_HoverMessage(self, msg);
}

void FG_FASTCALL fgList_Init(fgList* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgList_Destroy, (fgMessage)&fgList_Message);
}
void FG_FASTCALL fgList_Destroy(fgList* self)
{
  ((bss_util::cArraySort<fgElement*>&)self->selected).~cArraySort();
  fgBox_Destroy(&self->box);
}
void fgList_Draw(fgElement* self, const AbsRect* area, size_t dpi)
{
  fgList* realself = reinterpret_cast<fgList*>(self);
  for(size_t i = 0; i < realself->selected.l; ++i)
  {
    if(realself->selected.p[i]->GetClassName() != FGSTR_LISTITEM)
    {
      AbsRect r;
      ResolveRectCache(realself->selected.p[i], &r, area, (realself->selected.p[i]->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
      fgroot_instance->backend.fgDrawResource(0, &CRect { 0 }, realself->select.color, 0, 0.0f, &r, 0.0f, &AbsVec { 0,0 }, FGRESOURCE_ROUNDRECT);
    }
  }

  if(realself->mouse.state&FGMOUSE_DRAG)
  {
    AbsRect cache;
    fgElement* target = fgElement_GetChildUnderMouse(self, realself->mouse.x, realself->mouse.y, &cache);
    if(target)
    { // TODO: make this work with lists growing along x-axis
      AbsRect r;
      ResolveRectCache(target, &r, (AbsRect*)&cache, (target->flags & FGELEMENT_BACKGROUND) ? 0 : &self->padding);
      float y = (realself->mouse.y > ((r.top + r.bottom) * 0.5f)) ? r.bottom : r.top;
      AbsVec line[2] = { { r.left, y }, { r.right - 1, y } };
      fgroot_instance->backend.fgDrawLines(line, 2, realself->drag.color, &AbsVec { 0,0 }, &AbsVec { 1,1 }, 0, &AbsVec { 0,0 });
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
      fgroot_instance->backend.fgDrawResource(0, &CRect { 0 }, realself->hover.color, 0, 0.0f, &r, 0.0f, &AbsVec { 0,0 }, FGRESOURCE_ROUNDRECT);
    }
  }
}
size_t FG_FASTCALL fgList_Message(fgList* self, const FG_Msg* msg)
{
  ptrdiff_t otherint = msg->otherint;
  fgFlag flags = self->box->flags;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgBox_Message(&self->box, msg);
    memset(&self->selected, 0, sizeof(fgVectorElement));
    memset(&self->mouse, 0, sizeof(fgMouseState));
    self->box.fndraw = &fgList_Draw;
    self->select.color = 0xFF9999DD;
    self->hover.color = 0x99999999;
    self->drag.color = 0xFFCCCCCC;
    return FG_ACCEPT;
  case FG_MOUSEDOWN:
    fgUpdateMouseState(&self->mouse, msg);
    if(self->box->flags&FGLIST_SELECT)
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!target)
        break;
      size_t index = ((bss_util::cArraySort<fgElement*>&)self->selected).Find(target);
      if((self->box->flags&FGLIST_MULTISELECT) != FGLIST_MULTISELECT || !fgroot_instance->GetKey(FG_KEY_SHIFT))
      {
        for(size_t i = 0; i < self->selected.l; ++i)
          if(self->selected.p[i]->GetClassName() == FGSTR_LISTITEM)
            fgStandardNuetralSetStyle(self->selected.p[i], "selected", FGSETSTYLE_REMOVEFLAG);
        ((bss_util::cArraySort<fgElement*>&)self->selected).Clear();
      }
      else if(index != (size_t)-1)
      {
        if(self->selected.p[index]->GetClassName() == FGSTR_LISTITEM)
          fgStandardNuetralSetStyle(self->selected.p[index], "selected", FGSETSTYLE_REMOVEFLAG);
        ((bss_util::cArraySort<fgElement*>&)self->selected).Remove(index);
      }

      if(index == (size_t)-1)
      {
        if(target->GetClassName() == FGSTR_LISTITEM)
          fgStandardNuetralSetStyle(target, "selected", FGSETSTYLE_SETFLAG);
        ((bss_util::cArraySort<fgElement*>&)self->selected).Insert(target);
      }
    }
    break;
  case FG_MOUSEUP:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_MOUSEMOVE:
    fgUpdateMouseState(&self->mouse, msg);
    if((self->box->flags&FGLIST_DRAGGABLE) && (self->mouse.state&FGMOUSE_INSIDE)) // Check if we clicked inside this window
    {
      AbsRect cache;
      fgElement* target = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache); // find item below the mouse cursor (if any) and initiate a drag for it.
      if(target != 0)
      {
        fgroot_instance->backend.fgDragStart(FGCLIPBOARD_ELEMENT, target, target);
        fgCaptureWindow = 0;
        self->mouse.state &= ~FGMOUSE_INSIDE;
      }
    }
    break;
  case FG_MOUSEOFF:
    fgUpdateMouseState(&self->mouse, msg);
    break;
  case FG_DRAGOVER:
    fgUpdateMouseState(&self->mouse, msg);
    if(fgroot_instance->dragtype == FGCLIPBOARD_ELEMENT && fgroot_instance->dragdata != 0 && ((fgElement*)fgroot_instance->dragdata)->parent == *self) // Accept a drag element only if it's from this list
      fgRoot_SetCursor(FGCURSOR_DRAG, 0);
    else
      break; // the default handler rejects it for us
    return FG_ACCEPT;
  case FG_DROP:
    fgUpdateMouseState(&self->mouse, msg);
    if(fgroot_instance->dragtype == FGCLIPBOARD_ELEMENT && fgroot_instance->dragdata != 0)
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

      // TODO: figure out if we're on the x axis or y axis
      
      if(self->mouse.y > ((rect.top + rect.bottom) * 0.5f)) // if true, it's after target, so move the target pointer up one.
        target = target->next;

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
  case FG_SETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_SELECT:
    case FGSETCOLOR_MAIN: self->select.color = (unsigned int)msg->otherint; break;
    case FGSETCOLOR_HOVER: self->hover.color = (unsigned int)msg->otherint; break;
    case FGSETCOLOR_DRAG: self->drag.color = (unsigned int)msg->otherint; break;
    }
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return ((size_t)msg->otherint) < self->selected.l ? (size_t)self->selected.p[(size_t)msg->otherint] : 0;
  case FG_GETCLASSNAME:
    return (size_t)"List";
  }

  return fgBox_Message(&self->box, msg);
}