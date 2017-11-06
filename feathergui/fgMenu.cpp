// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgMenu.h"

static const char* SUBMENU_NAME = "Submenu";
static const char* MENU_NAME = "Menu";

void fgMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  assert(self != 0);
  const fgTransform TF_MENU = { { 0, 0, 0, 0, 0, 1.0, 0, 0 }, 0,{ 0,0,0,0 } };
  fgElement_InternalSetup(*self, parent, next, name, flags, (!transform ? &TF_MENU : transform), units, (fgDestroy)&fgMenu_Destroy, (fgMessage)&fgMenu_Message);
}

void fgMenu_Destroy(fgMenu* self)
{
  self->box->message = (fgMessage)fgBox_Message;
  fgBox_Destroy(&self->box);
  //fgRoot_DeallocAction(fgSingleton(),self->dropdown);
}

void fgMenu_Adjust(fgMenu* self, bool zero)
{
  if(zero)
  {
    CRect area = self->box->transform.area;
    area.bottom.abs -= area.top.abs;
    area.top.abs = 0;
    self->box->SetArea(area);
  }
  AbsRect out;
  ResolveRect(*self, &out);
  fgElement* e = (fgElement*)fgRoot_GetMonitor(fgroot_instance, &out);
  if(!e)
    e = &fgroot_instance->gui.element;
  AbsRect max;
  ResolveRect(e, &max);
  self->box->SetDim(self->box->GetDim(FGDIM_MAX)->x, max.bottom - max.top, FGDIM_MAX);
  FABS adjust = bssmin(0, max.bottom - out.bottom);
  if(adjust)
  {
    CRect area = self->box->transform.area;
    area.top.abs += adjust;
    area.bottom.abs += adjust;
    self->box->SetArea(area);
  }
}
void fgMenu_Show(fgMenu* self, bool show, bool zero)
{
  assert(self != 0);
  fgFlag set = show ? (self->box->flags & (~FGELEMENT_HIDDEN)) : (self->box->flags | FGELEMENT_HIDDEN);
  if(show)
    fgMenu_Adjust(self, zero);
  _sendmsg<FG_SETFLAGS, size_t>(*self, set);
  fgVoidMessage(self->box->parent, show ? FG_ACTIVE : FG_NEUTRAL, 0, 0);

  fgMenu* submenu = (fgMenu*)self->box->GetSelectedItem();
  if(submenu)
  {
    if(!show)
      self->expanded = 0;
    fgMenu_Show(submenu, show, true);
  }
  if(!show)
  {
    if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
      self->hover->Neutral();
    self->hover = 0;
  }
}

inline fgMenu* fgMenu_ExpandMenu(fgMenu* self, fgElement* child)
{
  fgMenu* submenu = reinterpret_cast<fgMenu*>(child->GetSelectedItem());
  if(child != self->hover)
  {
    if(self->expanded)
      fgMenu_Show(self->expanded, false, true);
    else if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
      self->hover->Neutral();
    self->expanded = (child->flags&FGCONTROL_DISABLE) ? 0 : submenu;
    self->hover = (child->flags&FGCONTROL_DISABLE) ? 0 : child;
    if(self->expanded)
      fgMenu_Show(self->expanded, true, true);
    else if(self->hover)
      self->hover->Active();
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
    self->hover = 0;
    self->arrow = 0;
    return FG_ACCEPT;
  }
  case FG_MOUSEUP:
    if(fgroot_instance->fgCaptureWindow == *self)
    {
      AbsRect cache;
      fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(!MsgHitAbsRect(msg, &cache))
      {
        if(self->expanded)
          fgMenu_Show(self->expanded, false, true);
        self->expanded = 0;
        fgroot_instance->fgCaptureWindow = 0;
      }
      else if(child != 0 && !fgMenu_ExpandMenu(self, child))
      {
        _sendmsg<FG_ACTION, void*>(*self, child);
        fgroot_instance->fgCaptureWindow = 0;
      }
    }
    return fgControl_Message((fgControl*)self, msg);
  case FG_MOUSEMOVE:
    if(fgroot_instance->fgCaptureWindow != *self)
    {
      AbsRect cache;
      fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
      if(child != self->hover)
      {
        if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
          self->hover->Neutral();
        self->hover = child;
        if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
          self->hover->Hover();
      }
      return fgControl_Message((fgControl*)self, msg);
    }
    break;
  case FG_MOUSEOFF:
    if(fgroot_instance->fgCaptureWindow != *self)
    {
      if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
        self->hover->Neutral();
      self->hover = 0;
    }
    break;
  case FG_MOUSEDOWN:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child != self->hover && self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
      self->hover->Neutral();
    self->hover = 0;
    if(fgroot_instance->fgCaptureWindow == *self)
    {
      if(self->expanded)
        fgMenu_Show(self->expanded, false, true);
      self->expanded = 0;
      if(self->hover && !(self->hover->flags&FGCONTROL_DISABLE))
        self->hover->Neutral();
      self->hover = 0;
      fgroot_instance->fgCaptureWindow = 0;
      return fgControl_Message((fgControl*)self, msg);
    }
    fgroot_instance->fgCaptureWindow = *self;
    if(child != 0)
      fgMenu_ExpandMenu(self, child);
  }
    return fgControl_Message((fgControl*)self, msg);
  case FG_ADDITEM:
    switch(msg->subtype)
    {
    case FGITEM_DEFAULT:
      break;
    case FGITEM_ELEMENT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
      fgSendMessage(menuitem, msg);
      return (size_t)menuitem;
    }
    case FGITEM_TEXT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
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

void fgSubmenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  assert(self != 0);
  const fgTransform TF_MENU = { { 0, 0, 0, 1.0, 0, 0, 0, 1.0 }, 0,{ 0,0,0,0 } };
  const fgTransform TF_SUBMENU = { { 0, 1.0, 0, 0, 0, 1.0, 0, 0 }, 0,{ 0,0,0,0 } };
  if(!transform) transform = (parent != 0 && parent->parent != 0 && parent->parent->GetClassName() == MENU_NAME) ? &TF_MENU : &TF_SUBMENU;
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgMenu_Destroy, (fgMessage)&fgSubmenu_Message);
}
void fgContextMenu_Init(fgMenu* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  assert(self != 0);
  if(!transform) transform = &fgTransform_EMPTY;
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgMenu_Destroy, (fgMessage)&fgSubmenu_Message);
}

size_t fgSubmenu_Message(fgMenu* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
  {
    fgBox_Message(&self->box, msg);
    self->expanded = 0;
    self->hover = 0;
    return FG_ACCEPT;
  }
  case FG_CLONE:
    if(msg->e)
    {
      fgMenu* hold = reinterpret_cast<fgMenu*>(msg->e);
      hold->expanded = 0;
      hold->hover = 0;
      fgBox_Message(&self->box, msg);
    }
    return sizeof(fgMenu);
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = bss::bssSetBit<fgFlag>(self->box->flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if(((self->box->flags ^ otherint) & FGELEMENT_HIDDEN) != 0)
    {
      if(!(otherint&FGELEMENT_HIDDEN) && self->box->parent != 0 && self->box->parent->parent != 0 && self->box->parent->parent->GetClassName() != SUBMENU_NAME)
        fgSetTopmost(self->box);
      else
        fgClearTopmost(self->box);
    }
    break;
  case FG_MOUSEUP:
    msg = msg;
  case FG_MOUSEDOWN:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(MsgHitAbsRect(msg, &cache))
    {
      if(child != 0 && !fgMenu_ExpandMenu(self, child) && msg->type == FG_MOUSEUP)
        _sendmsg<FG_ACTION, void*>(*self, child);
      else
        return FG_ACCEPT;
    }
    if((fgroot_instance->fgCaptureWindow != 0) && (fgroot_instance->fgCaptureWindow == *self || fgroot_instance->fgCaptureWindow->GetClassName() == SUBMENU_NAME))
    {
      fgMenu_Show((fgMenu*)fgroot_instance->fgCaptureWindow, false, true);
      fgroot_instance->fgCaptureWindow = 0;
    }
    if(fgroot_instance->fgCaptureWindow != 0 && (fgroot_instance->fgCaptureWindow->GetClassName() == MENU_NAME))
    {
      fgMenu* menu = reinterpret_cast<fgMenu*>(fgroot_instance->fgCaptureWindow);
      if(menu->expanded)
        fgMenu_Show(menu->expanded, false, true);
      menu->expanded = 0;
      if(menu->hover)
        menu->hover->Neutral();
      menu->hover = 0;
      fgroot_instance->fgCaptureWindow = 0;
    }
  }
  return FG_ACCEPT;
  case FG_MOUSEMOVE:
  {
    AbsRect cache;
    fgElement* child = fgElement_GetChildUnderMouse(*self, msg->x, msg->y, &cache);
    if(child != 0)
      fgMenu_ExpandMenu(self, child);
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
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
      fgSendMessage(menuitem, msg);
      return (size_t)menuitem;
    }
    case FGITEM_TEXT:
    {
      fgElement* menuitem = fgroot_instance->backend.fgCreate("MenuItem", *self, 0, 0, FGFLAGS_DEFAULTS, 0, 0);
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
          ResolveRectCache(cur, &rect, (const AbsRect*)msg->p, 0);
          fgDrawSkin(self->arrow, &rect, (fgDrawAuxData*)msg->p2, false);
        }
        cur = cur->next;
      }
    }
    return FG_ACCEPT;
  case FG_SETSKIN:
    fgBox_Message(&self->box, msg);
    if(self->box->skin)
      self->arrow = self->box->skin->base.GetSkin("Submenu$arrow");
    return FG_ACCEPT;
  case FG_GETSELECTEDITEM:
    return (size_t)self->expanded;
  case FG_GETCLASSNAME:
    return (size_t)SUBMENU_NAME; // This allows you to properly differentiate a top level menu glued to the top of a window from a submenu, like a context menu.
  }
  return fgBox_Message(&self->box, msg);
}

void fgMenuItem_Init(fgMenuItem* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  static const fgTransform MENU_TRANSFORM = fgTransform { { 0,0,0,0,0,1,0,0 }, 0, { 0,0,0,0 } };
  if(!transform) transform = (parent != 0 && parent->GetClassName() == MENU_NAME) ? &fgTransform_EMPTY : &MENU_TRANSFORM;
  assert(self != 0);
  fgElement_InternalSetup(&self->element, parent, next, name, flags, transform, units, (fgDestroy)&fgElement_Destroy, (fgMessage)&fgMenuItem_Message);
}

size_t fgMenuItem_Message(fgMenuItem* self, const FG_Msg* msg)
{
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(&self->element, msg);
    fgText_Init(&self->text, &self->element, 0, "MenuItem$text", FGELEMENT_EXPAND | FGFLAGS_INTERNAL, 0, 0);
    self->submenu = 0;
    self->element.SetStyle((self->element.flags & FGCONTROL_DISABLE) ? "disabled" : "neutral");
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgMenuItem* hold = reinterpret_cast<fgMenuItem*>(msg->e);
      hold->submenu = 0;
      self->text->Clone(hold->text);
      fgElement_Message(&self->element, msg);
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->text);
    }
    return sizeof(fgMenuItem);
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
  case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
    otherint = bss::bssSetBit<fgFlag>(self->element.flags, otherint, msg->u2 != 0);
  case FG_SETFLAGS:
    if((self->element.flags ^ (fgFlag)otherint) & FGCONTROL_DISABLE)
    {
      self->element.SetStyle((otherint & FGCONTROL_DISABLE) ? "disabled" : "neutral");
      fgroot_instance->mouse.state |= FGMOUSE_SEND_MOUSEMOVE;
    }
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
    return fgSendMessage(self->text, msg);
  case FG_GETCLASSNAME:
    return (size_t)"MenuItem";
  }
  return fgElement_Message(&self->element, msg);
}
