// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgMenu.h"
#include "feathercpp.h"

static const char* SUBMENU_NAME = "Submenu";
static const char* MENU_NAME = "Menu";

void fgMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  assert(self != 0);
  const fgTransform TF_MENU = { { 0, 0, 0, 0, 0, 1.0, 0, 0 }, 0,{ 0,0,0,0 } };
  fgElement_InternalSetup(*self, parent, next, name, ((flags&FGELEMENT_USEDEFAULTS) ? (FGELEMENT_EXPANDY | FGBOX_TILEX) : flags), (!transform ? &TF_MENU : transform), units, (fgDestroy)&fgMenu_Destroy, (fgMessage)&fgMenu_Message);
}

void fgMenu_Destroy(fgMenu* self)
{
  fgElement_Destroy(&self->arrow); // This must be destroyed manually because it's a ghost entity not attached to the menu.
  self->box->message = (fgMessage)fgBox_Message;
  fgBox_Destroy(&self->box);
  //fgRoot_DeallocAction(fgSingleton(),self->dropdown);
}

void fgMenu_Show(fgMenu* self, bool show)
{
  assert(self != 0);
  fgFlag set = show ? (self->box->flags & (~FGELEMENT_HIDDEN)) : (self->box->flags | FGELEMENT_HIDDEN);
  _sendmsg<FG_SETFLAGS, size_t>(*self, set);

  fgMenu* submenu = (fgMenu*)self->box->GetSelectedItem();
  if(submenu)
  {
    if(!show)
      self->expanded = 0;
    fgMenu_Show(submenu, show);
  }
}

inline fgMenu* fgMenu_ExpandMenu(fgMenu* self, fgMenu* submenu)
{
  if(submenu != self->expanded)
  {
    if(self->expanded)
      fgMenu_Show(self->expanded, false);
    self->expanded = submenu;
    if(self->expanded)
      fgMenu_Show(self->expanded, true);
  }

  return submenu;
}
size_t fgMenu_Message(fgMenu* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    fgBox_Message(&self->box, msg);
    self->expanded = 0;
    const fgTransform TF_ARROW = { { 0,0,0,0.5,0,0,0,0.5 }, 0,{ 1.0,0.5 } };
    fgElement_Init(&self->arrow, 0, 0, "Menu$arrow", FGELEMENT_IGNORE | FGELEMENT_BACKGROUND | FGELEMENT_EXPAND, &TF_ARROW, 0);
    return FG_ACCEPT;
  }
  case FG_MOUSEUP:
    if(fgCaptureWindow == *self)
    {
      AbsRect cache;
      fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!MsgHitAbsRect(msg, &cache))
        fgCaptureWindow = 0;
      else if(child != 0 && !fgMenu_ExpandMenu(self, (fgMenu*)child->GetSelectedItem()))
      {
        _sendmsg<FG_ACTION, void*>(*self, child);
        fgCaptureWindow = 0;
      }
    }
    return fgControl_Message((fgControl*)self, msg);
  case FG_MOUSEMOVE:
    if(fgCaptureWindow != *self)
      return fgControl_Message((fgControl*)self, msg);
    break;
  case FG_MOUSEDOWN:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(fgCaptureWindow == *self)
    {
      if(self->expanded)
        fgMenu_Show(self->expanded, false);
      self->expanded = 0;
      fgCaptureWindow = 0;
      return fgControl_Message((fgControl*)self, msg);
    }
    fgCaptureWindow = *self;
    if(child != 0)
      fgMenu_ExpandMenu(self, (fgMenu*)child->GetSelectedItem());
  }
    return fgControl_Message((fgControl*)self, msg);
  case FG_ADDITEM:
    switch(msg->subtype)
    {
    case FGITEM_DEFAULT:
      break;
    case FGITEM_ELEMENT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
      fgPassMessage(menuitem, msg);
      return (size_t)menuitem;
    }
    case FGITEM_TEXT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
      menuitem->SetText((const char*)msg->p, (FGTEXTFMT)msg->u2);
      return (size_t)menuitem;
    }
    }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)MENU_NAME;
  case FG_DRAW:
    return fgBox_Message(&self->box, msg);
  }
  return fgSubmenu_Message(self, msg);
}

void fgSubmenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  assert(self != 0);
  const fgTransform TF_MENU = { { 0, 0, 0, 1.0, 0, 0, 0, 1.0 }, 0,{ 0,0,0,0 } };
  const fgTransform TF_SUBMENU = { { 0, 1.0, 0, 0, 0, 1.0, 0, 0 }, 0,{ 0,0,0,0 } };
  if(!transform) transform = (parent != 0 && parent->parent != 0 && parent->parent->GetClassName() == MENU_NAME) ? &TF_MENU : &TF_SUBMENU;
  fgElement_InternalSetup(*self, parent, next, name, ((flags&FGELEMENT_USEDEFAULTS) ? (FGELEMENT_NOCLIP | FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGBOX_TILEY | FGELEMENT_EXPAND) : flags), transform, units, (fgDestroy)&fgMenu_Destroy, (fgMessage)&fgSubmenu_Message);
}

size_t fgSubmenu_Message(fgMenu* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    fgBox_Message(&self->box, msg);
    self->expanded = 0;
    const fgTransform TF_ARROW = { { 0,1,0,0.5,0,1,0,0.5 }, 0,{ 0.5,1.0 } };
    fgElement_Init(&self->arrow, 0, 0, "Submenu$arrow", FGELEMENT_IGNORE | FGELEMENT_BACKGROUND | FGELEMENT_EXPAND, &TF_ARROW, 0);
    return FG_ACCEPT;
  }
  case FG_MOUSEUP:
  case FG_MOUSEDOWN:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(MsgHitAbsRect(msg, &cache))
    {
      if(child != 0 && !fgMenu_ExpandMenu(self, (fgMenu*)child->GetSelectedItem()) && msg->type == FG_MOUSEUP)
        _sendmsg<FG_ACTION, void*>(*self, child);
      else
        return FG_ACCEPT;
    }
    if((fgCaptureWindow != 0) && (fgCaptureWindow == *self || fgCaptureWindow->GetClassName() == SUBMENU_NAME))
    {
      fgMenu_Show((fgMenu*)fgCaptureWindow, false);
      fgCaptureWindow = 0;
    }
    if(fgCaptureWindow != 0 && (fgCaptureWindow->GetClassName() == MENU_NAME))
    {
      fgMenu* menu = reinterpret_cast<fgMenu*>(fgCaptureWindow);
      if(menu->expanded)
        fgMenu_Show(menu->expanded, false);
      menu->expanded = 0;
      fgCaptureWindow = 0;
    }
  }
  return FG_ACCEPT;
  case FG_MOUSEMOVE:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child != 0)
      fgMenu_ExpandMenu(self, (fgMenu*)child->GetSelectedItem());
  }
  break;
  case FG_ADDITEM:
    if(!msg->p)
    {
      fgElement* item = fgmalloc<fgElement>(1, __FILE__, __LINE__);
      const fgTransform TF_SEPERATOR = { { 0,0,0,0,0,1.0,0,0 }, 0,{ 0,0 } };
      fgElement_Init(item, *self, 0, "Submenu$seperator", FGELEMENT_IGNORE, &TF_SEPERATOR, 0);
      item->free = &fgfreeblank;
      return (size_t)item;
    }
    switch(msg->subtype)
    {
    case FGITEM_DEFAULT:
      break;
    case FGITEM_ELEMENT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
      fgPassMessage(menuitem, msg);
      return (size_t)menuitem;
    }
    case FGITEM_TEXT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGELEMENT_USEDEFAULTS, 0, 0);
      menuitem->SetText((const char*)msg->p, (FGTEXTFMT)msg->u2);
      return (size_t)menuitem;
    }
    }
    return 0;
  case FG_DRAW:
    fgBox_Message(&self->box, msg);
    if(!(msg->subtype & 1))
    {
      fgElement* cur = self->box->root;
      while(cur)
      {
        if(!(cur->flags & FGELEMENT_BACKGROUND) && cur->GetSelectedItem())
        {
          AbsRect rect;
          ResolveRectCache(cur, &rect, (const AbsRect*)msg->p, &self->box->padding);
          AbsRect arrow;
          ResolveRectCache(&self->arrow, &arrow, &rect, (self->arrow.flags&FGELEMENT_BACKGROUND) ? 0 : &cur->padding);
          self->arrow.Draw(&arrow, (fgDrawAuxData*)msg->p2);
        }
        cur = cur->next;
      }
    }
    return FG_ACCEPT;
  case FG_SETSKIN:
    fgBox_Message(&self->box, msg);
    self->arrow.SetSkin(self->box->GetSkin(&self->arrow)); // Because the arrow is not technically a child of the menu, we must set the skin manually
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return (size_t)self->expanded;
  case FG_GETCLASSNAME:
    return (size_t)SUBMENU_NAME; // This allows you to properly differentiate a top level menu glued to the top of a window from a submenu, like a context menu.
  }
  return fgBox_Message(&self->box, msg);
}

void fgMenuItem_Init(fgMenuItem* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  static const fgTransform MENU_TRANSFORM = fgTransform { { 0,0,0,0,0,1,0,0 }, 0, { 0,0,0,0 } };
  if(!transform) transform = (parent != 0 && parent->GetClassName() == MENU_NAME) ? &fgTransform_EMPTY : &MENU_TRANSFORM;
  if(flags&FGELEMENT_USEDEFAULTS) flags = FGELEMENT_EXPAND | FGELEMENT_NOCLIP;
  assert(self != 0);
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, units, (fgDestroy)&fgElement_Destroy, (fgMessage)&fgMenuItem_Message);
}

size_t fgMenuItem_Message(fgMenuItem* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(&self->element, msg);
    fgText_Init(&self->text, &self->element, 0, "MenuItem$text", FGELEMENT_EXPAND, 0, 0);
    self->submenu = 0;
    return FG_ACCEPT;
  case FG_ADDITEM:
  case FG_ADDCHILD:
  {
    size_t r = fgElement_Message(&self->element, msg);
    if(r != 0 && msg->e->GetClassName() == SUBMENU_NAME)
      self->submenu = (fgMenu*)msg->p;
    return r;
  }
  case FG_SETITEM:
    if(msg->subtype == FGITEM_TEXT)
      return self->text->SetText((const char*)msg->p, (FGTEXTFMT)msg->u2);
    break;
  case FG_GETSELECTEDITEM:
    return (size_t)self->submenu;
  case FG_SETTEXT:
  case FG_SETFONT:
  case FG_SETLINEHEIGHT:
  case FG_SETLETTERSPACING:
  case FG_SETCOLOR:
  case FG_GETTEXT:
  case FG_GETFONT:
  case FG_GETLINEHEIGHT:
  case FG_GETLETTERSPACING:
  case FG_GETCOLOR:
    return fgPassMessage(self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"MenuItem";
  }
  return fgElement_Message(&self->element, msg);
}
