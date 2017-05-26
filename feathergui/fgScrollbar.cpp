// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util/bss_util.h"
#include "feathercpp.h"

void fgScrollbar_Init(fgScrollbar* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgScrollbar_Destroy, (fgMessage)&fgScrollbar_Message);
}
void fgScrollbar_Destroy(fgScrollbar* self)
{
  self->control->message = (fgMessage)fgControl_Message;
  fgControl_Destroy(&self->control);
}

size_t fgScrollbar_cornerMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
  case FG_MOUSEMOVE:
    fgElement_DoHoverCalc(self);
  case FG_MOUSEUP:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
    return FG_ACCEPT;
  }
  return fgElement_Message(self, msg);
}

size_t fgScrollbar_bgMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(self, msg);
    _sendsubmsg<FG_ACTION>(self->parent, FGSCROLLBAR_CHANGE);
    return FG_ACCEPT;
  case FG_ACTION:
    if(self->parent != 0)
      return fgSendMessage(self->parent, msg);
    break;
  case FG_MOUSEDOWN:
    if(msg->button == FG_MOUSELBUTTON && self->parent != 0)
    {
      fgScrollbar* parent = (fgScrollbar*)self->parent;
      AbsRect r;
      ResolveRect(parent->bar[self->userid].button, &r);
      if(!self->userid)
        _sendsubmsg<FG_ACTION, ptrdiff_t>(self->parent, FGSCROLLBAR_PAGE, msg->x < r.left ? 0 : 2);
      else
        _sendsubmsg<FG_ACTION, ptrdiff_t>(self->parent, FGSCROLLBAR_PAGE, msg->y < r.top ? 1 : 3);
    }
  case FG_MOUSEMOVE:
    fgElement_DoHoverCalc(self);
  case FG_MOUSEUP:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
    return FG_ACCEPT;
  case FG_SETAREA:
  {
    size_t ret = fgElement_Message(self, msg);
    if(ret != 0 && self->parent != 0)
      _sendsubmsg<FG_ACTION>(self->parent, FGSCROLLBAR_CHANGE);
    return ret;
  }
  case FG_MOUSESCROLL:
    return 0;
  }

  return fgElement_Message(self, msg);
}

size_t fgScrollbar_barMessage(_FG_SCROLLBAR_INNER* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgButton_Message(&self->button, msg);
    self->lastmouse.x = 0;
    self->lastmouse.y = 0;
    return FG_ACCEPT;
  case FG_MOUSEUP:
  case FG_MOUSEDOWN:
    self->lastmouse.x = (FABS)msg->x;
    self->lastmouse.y = (FABS)msg->y;
    if(self->button->parent != 0)
      _sendsubmsg<FG_ACTION, ptrdiff_t>(self->button->parent, FGSCROLLBAR_BARINIT, self->button->userid);
    break;
  case FG_MOUSEMOVE:
    if(fgroot_instance->fgCaptureWindow == self->button)
    {
      if(self->button->parent != 0)
        _sendsubmsg<FG_ACTION, ptrdiff_t, float>(
          self->button->parent,
          FGSCROLLBAR_BAR,
          self->button->userid,
          (!self->button->userid) ? (self->lastmouse.x - (FABS)msg->x) : (self->lastmouse.y - (FABS)msg->y));
    }
    break;
  case FG_MOUSESCROLL:
    return 0;
  }
  return fgButton_Message(&self->button, msg);
}

size_t fgScrollbar_buttonMessage(fgButton* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
    if(self->control.element.parent != 0)
      _sendsubmsg<FG_ACTION, ptrdiff_t>(self->control.element.parent, FGSCROLLBAR_BUTTON, self->control.element.userid);
    break;
  case FG_MOUSESCROLL:
    return 0;
  }
  return fgButton_Message(self, msg);
}

void fgScrollbar_ApplyPadding(fgScrollbar* self, float x, float y);

void fgScrollbar_Recalc(fgScrollbar* self)
{
  AbsRect r;
  ResolveRect(*self, &r);
  AbsVec dim = { r.right - r.left, r.bottom - r.top };

  bool hideh = !!(self->control.element.flags&FGSCROLLBAR_HIDEH);
  bool hidev = !!(self->control.element.flags&FGSCROLLBAR_HIDEV);

  // We have to figure out which scrollbars are visible based on flags and our dimensions. We use a half-pixel buffer so precision errors don't toggle scrollbars.
  bool scrollx = !hideh && ((dim.x < self->realsize.x + self->realpadding.left + self->realpadding.right + self->exclude.left + self->exclude.right - 0.5f) || self->control.element.flags&FGSCROLLBAR_SHOWH);
  bool scrolly = !hidev && ((dim.y < self->realsize.y + self->realpadding.top + self->realpadding.bottom + self->exclude.top + self->exclude.bottom - 0.5f) || self->control.element.flags&FGSCROLLBAR_SHOWV);

  // If we are adding or removing a scrollbar, this will change the padding and could change everything else (e.g. for textboxes)
  char dimr = 0;
  if(scrollx ^ ((self->bg[0].flags&FGELEMENT_HIDDEN) == 0))
  {
    if(scrollx)
    {
      ResolveRect(&self->bg[0], &r);
      self->exclude.bottom = r.bottom - r.top;
    }
    else
      self->exclude.bottom = 0;
    dimr |= FGMOVE_RESIZEY;
  }
  if(scrolly ^ ((self->bg[1].flags&FGELEMENT_HIDDEN) == 0))
  {
    if(scrolly)
    {
      ResolveRect(&self->bg[1], &r);
      self->exclude.right = r.right - r.left;
    }
    else
      self->exclude.right = 0;
    dimr |= FGMOVE_RESIZEY;
  }
  if(dimr)
  {
    self->bg[0].SetFlag(FGELEMENT_HIDDEN, !scrollx);
    self->bg[1].SetFlag(FGELEMENT_HIDDEN, !scrolly);
    fgScrollbar_ApplyPadding(self, 0, 0);
    self->control->Move(FG_SETPADDING, 0, dimr);
    return;
  }

  // Expand area in response to adding scrollbars if necessary
  CRect area = self->control.element.transform.area;
  if(self->control.element.flags & FGELEMENT_EXPANDX)
    area.right.abs = area.left.abs + self->realsize.x + self->realpadding.left + self->realpadding.right + self->control->margin.left + self->control->margin.right + self->exclude.left + self->exclude.right;
  if(self->control.element.flags & FGELEMENT_EXPANDY)
    area.bottom.abs = area.top.abs + self->realsize.y + self->realpadding.top + self->realpadding.bottom + self->control->margin.top + self->control->margin.bottom + self->exclude.top + self->exclude.bottom;
  _sendmsg<FG_SETAREA, void*>(*self, &area); // SETAREA will set MAXDIM appropriately

  // If both scrollbars are active, set the margins to exclude them from the corner
  self->bg[0].SetMargin(AbsRect { self->exclude.left,0,self->exclude.right,0 });
  self->bg[1].SetMargin(AbsRect { 0,self->exclude.top,0, self->exclude.bottom });
  if(self->exclude.right > 0 && self->exclude.bottom > 0)
  {
    self->bg[2].SetFlag(FGELEMENT_HIDDEN | FGELEMENT_IGNORE, false);
    self->bg[2].SetArea(CRect{ -self->exclude.right, 1, -self->exclude.bottom, 1, 0, 1, 0, 1 });
  }
  else
    self->bg[2].SetFlag(FGELEMENT_HIDDEN | FGELEMENT_IGNORE, true);

  // Set bar dimensions appropriately based on padding
  if(scrollx)
  {
    float d = r.right - r.left - self->realpadding.left - self->realpadding.right - self->exclude.left - self->exclude.right;
    float length = (self->realsize.x <= 0.0f) ? 1.0f : (d / self->realsize.x);
    float pos = (d - self->realsize.x) == 0.0f ? 0.0f : (self->control.element.padding.left - self->realpadding.left - self->exclude.left) / (d - self->realsize.x);
    self->bar[0].button->SetArea(CRect { 0, (1.0f - length)*pos, 0, 0, 0, (1.0f - length)*pos + length, 0, 1.0f });
    self->btn[0]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos <= 0.0f);
    self->btn[2]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos >= 1.0f);
    self->bar[0].button->SetFlag(FGCONTROL_DISABLE, length >= 1.0f);
  }
  if(scrolly)
  {
    float d = r.bottom - r.top - self->realpadding.top - self->realpadding.bottom - self->exclude.top - self->exclude.bottom;
    float length = (self->realsize.y <= 0.0f) ? 1.0f : (d / self->realsize.y);
    float pos = (d - self->realsize.y) == 0.0f ? 0.0f : (self->control.element.padding.top - self->realpadding.top - self->exclude.top) / (d - self->realsize.y);
    self->bar[1].button->SetArea(CRect { 0, 0, 0, (1.0f - length)*pos, 0, 1.0f, 0, (1.0f - length)*pos + length });
    self->btn[1]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos <= 0.0f);
    self->btn[3]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos >= 1.0f);
    self->bar[1].button->SetFlag(FGCONTROL_DISABLE, length >= 1.0f);
  }
}

void fgScrollbar_ApplyPadding(fgScrollbar* self, float x, float y)
{
  AbsRect totalpadding = {
    self->realpadding.left + self->exclude.left,
    self->realpadding.top + self->exclude.top,
    self->realpadding.right + self->exclude.right,
    self->realpadding.bottom + self->exclude.bottom
  };

  assert(!std::isnan(self->realsize.x) && !std::isnan(self->realsize.y));
  self->control.element.padding.left += x;
  self->control.element.padding.top += y;
  if(self->control.element.padding.left > totalpadding.left)
    self->control.element.padding.left = totalpadding.left;
  if(self->control.element.padding.top > totalpadding.top)
    self->control.element.padding.top = totalpadding.top;

  AbsRect r;
  ResolveRect(*self, &r);
  float minpaddingx = (r.right - r.left - totalpadding.right) - self->realsize.x;
  minpaddingx = bssmin(minpaddingx, totalpadding.left);
  float minpaddingy = (r.bottom - r.top - totalpadding.bottom) - self->realsize.y;
  minpaddingy = bssmin(minpaddingy, totalpadding.top);
  if(self->control.element.padding.left < minpaddingx)
    self->control.element.padding.left = minpaddingx;
  if(self->control.element.padding.top < minpaddingy)
    self->control.element.padding.top = minpaddingy;

  self->control.element.padding.right = totalpadding.right;
  self->control.element.padding.bottom = totalpadding.bottom;

  fgScrollbar_Recalc(self); // recalculate scrollbars from our new padding value
}

size_t fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  static const float DEFAULT_LINEHEIGHT = 30.0f;
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    bss::bssFill(self->realpadding, 0);
    bss::bssFill(self->exclude, 0);
    bss::bssFill(self->realsize, 0);
    fgControl_Message(&self->control, msg);
    fgElement_Init(&self->bg[0], *self, 0, "Scrollbar$horzbg", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGFLAGS_INTERNAL, 0, 0);
    fgElement_Init(&self->bg[1], *self, 0, "Scrollbar$vertbg", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGFLAGS_INTERNAL, 0, 0);
    fgElement_Init(&self->bg[2], *self, 0, "Scrollbar$corner", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->btn[0], &self->bg[0], 0, "Scrollbar$scrollleft", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->btn[1], &self->bg[1], 0, "Scrollbar$scrolltop", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->btn[2], &self->bg[0], 0, "Scrollbar$scrollright", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->btn[3], &self->bg[1], 0, "Scrollbar$scrollbottom", FGELEMENT_BACKGROUND | FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->bar[0].button, &self->bg[0], 0, "Scrollbar$scrollhorz", FGFLAGS_INTERNAL, 0, 0);
    fgButton_Init(&self->bar[1].button, &self->bg[1], 0, "Scrollbar$scrollvert", FGFLAGS_INTERNAL, 0, 0);
    self->btn[0]->userid = 0;
    self->btn[1]->userid = 1;
    self->btn[2]->userid = 2;
    self->btn[3]->userid = 3;
    self->bar[0].button->userid = 0;
    self->bar[1].button->userid = 1;
    self->bg[0].userid = 0;
    self->bg[1].userid = 1;
    self->bg[2].userid = 2;
    self->bg[0].message = (fgMessage)&fgScrollbar_bgMessage;
    self->bg[1].message = (fgMessage)&fgScrollbar_bgMessage;
    self->bg[2].message = (fgMessage)&fgScrollbar_cornerMessage;
    self->btn[0].control.element.message = (fgMessage)&fgScrollbar_buttonMessage;
    self->btn[1].control.element.message = (fgMessage)&fgScrollbar_buttonMessage;
    self->btn[2].control.element.message = (fgMessage)&fgScrollbar_buttonMessage;
    self->btn[3].control.element.message = (fgMessage)&fgScrollbar_buttonMessage;
    self->bar[0].button.control.element.message = (fgMessage)&fgScrollbar_barMessage;
    self->bar[1].button.control.element.message = (fgMessage)&fgScrollbar_barMessage;
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgScrollbar* hold = reinterpret_cast<fgScrollbar*>(msg->e);
      hold->realpadding = self->realpadding;
      hold->exclude = self->exclude;
      hold->realsize = self->realsize;
      hold->lastpadding = self->lastpadding;
      fgControl_Message(&self->control, msg);
      for(size_t i = 0; i < 4; ++i) { self->btn[i]->Clone(hold->btn[i]); _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->btn[i]); }
      for(size_t i = 0; i < 3; ++i) { self->bg[i].Clone(&hold->bg[i]); _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, &hold->bg[i]); }
      self->bar[0].button->Clone(hold->bar[0].button);
      hold->bar[0].lastmouse = self->bar[0].lastmouse;
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->bar[0].button);
      self->bar[1].button->Clone(hold->bar[1].button);
      hold->bar[1].lastmouse = self->bar[1].lastmouse;
      _sendmsg<FG_ADDCHILD, fgElement*>(msg->e, hold->bar[1].button);
    }
    return sizeof(fgScrollbar);
  case FG_SETPADDING:
    if(msg->p != nullptr)
    {
      AbsRect* padding = (AbsRect*)msg->p;
      char diff = CompareMargins(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff) // Only send a move message if the padding change actually changed something
        fgSubMessage(*self, FG_MOVE, FG_SETPADDING, 0, FGMOVE_PADDING);

      fgScrollbar_ApplyPadding(self, 0, 0); // Recalculate scrollbar positions
      return FG_ACCEPT;
    }
    fgLog(FGLOG_INFO, "%s set NULL padding.", fgGetFullName(*self).c_str(), msg->subtype);
    return 0;
  case FG_MOUSESCROLL:
  {
    FG_Msg m { 0 };
    m.type = FG_ACTION;
    m.subtype = FGSCROLLBAR_CHANGE;
    float lineheight = self->control.element.GetLineHeight();
    m.f2 = (msg->scrolldelta / 120.0f) * 3.0f * lineheight;
    m.f = (msg->scrollhdelta / -120.0f) * lineheight;

    AbsRect r;
    ResolveRect(*self, &r);
    m.f2 = std::min<FABS>(abs(m.f2), r.bottom - r.top - self->exclude.top - self->exclude.bottom) * (std::signbit(m.f2) ? -1.0f : 1.0f);

    if((m.f2 == 0.0f || (self->bg[1].flags&FGELEMENT_HIDDEN)) && (m.f == 0.0f || (self->bg[0].flags&FGELEMENT_HIDDEN)))
      return 0; // If there is no scrolling to be done, reject this message
    fgSendMessage(*self, &m);
    return FG_ACCEPT;
  }
  case FG_LAYOUTCHANGE:
    { 
      AbsVec oldsize = self->realsize;
      _sendsubmsg<FG_LAYOUTFUNCTION, const void*, void*>(*self, 1, msg, &self->realsize);
      assert(!std::isnan(self->realsize.x) && !std::isnan(self->realsize.y));
      if(oldsize.x != self->realsize.x || oldsize.y != self->realsize.y || (msg->u2&FGMOVE_RESIZE))
      {
        if(self->control.element.flags & FGELEMENT_EXPAND) // We only need to adjust the actual area if we are expanding. Otherwise we just want to update the padding to the new realsize.
        {
          CRect area = self->control.element.transform.area;
          if(self->control.element.flags & FGELEMENT_EXPANDX)
            area.right.abs = area.left.abs + self->realsize.x + self->realpadding.left + self->realpadding.right + self->control->margin.left + self->control->margin.right + self->exclude.left + self->exclude.right;
          if(self->control.element.flags & FGELEMENT_EXPANDY)
            area.bottom.abs = area.top.abs + self->realsize.y + self->realpadding.top + self->realpadding.bottom + self->control->margin.top + self->control->margin.bottom + self->exclude.top + self->exclude.bottom;
          self->control.element.SetArea(area);
        }
        fgScrollbar_ApplyPadding(self, 0, 0); // we must do applypadding here so the scroll area responds correctly when realsize shrinks.
      }
    }
    return FG_ACCEPT;
  case FG_GETLINEHEIGHT:
  {
    size_t r = fgControl_HoverMessage(&self->control, msg);
    return (*reinterpret_cast<float*>(&r) == 0.0f) ? *reinterpret_cast<const size_t*>(&DEFAULT_LINEHEIGHT) : r;
  }
  case FG_ACTION:
    switch(msg->subtype)
    {
    case FGSCROLLBAR_BARINIT:
      if(!msg->i) self->lastpadding.x = self->control.element.padding.left;
      else self->lastpadding.y = self->control.element.padding.top;
      return FG_ACCEPT;
    case FGSCROLLBAR_BAR:
    {
      AbsRect r;
      ResolveRect(&self->bg[msg->i != 0], &r);
      if(!msg->i)
        fgScrollbar_ApplyPadding(self, self->lastpadding.x + msg->f2 * self->realsize.x / (r.right - r.left - self->realpadding.left - self->realpadding.right - self->bg[0].padding.left - self->bg[0].padding.right) - self->control.element.padding.left, 0);
      else
        fgScrollbar_ApplyPadding(self, 0, self->lastpadding.y + msg->f2 * self->realsize.y / (r.bottom - r.top - self->realpadding.top - self->realpadding.bottom - self->bg[1].padding.top - self->bg[1].padding.bottom) - self->control.element.padding.top);
    }
      return FG_ACCEPT;
    case FGSCROLLBAR_PAGE: // By default a page scroll (clicking on the area outside of the bar or hitting pageup/pagedown) attempts to scroll by the width of the container in that direction.
    {
      AbsRect r;
      ResolveRect(*self, &r);
      if(msg->i&1)
        fgScrollbar_ApplyPadding(self, 0, (r.bottom - r.top)*(msg->i == 1 ? 1 : -1));
      else
        fgScrollbar_ApplyPadding(self, (r.right - r.left)*(msg->i == 0 ? 1 : -1), 0);
    }
      return FG_ACCEPT;
    case FGSCROLLBAR_BUTTON:
    {
      float x = msg->i < 2 ? self->control.element.GetLineHeight() : -self->control.element.GetLineHeight();
      if(msg->i % 2)
        fgScrollbar_ApplyPadding(self, 0, x);
      else
        fgScrollbar_ApplyPadding(self, x, 0);
      return FG_ACCEPT;
    }
    case FGSCROLLBAR_CHANGE: // corresponds to an actual change amount. First argument is x axis, second argument is y axis.
      fgScrollbar_ApplyPadding(self, msg->f, msg->f2);
      return FG_ACCEPT;
    case FGSCROLLBAR_SCROLLTO:
    case FGSCROLLBAR_SCROLLTOABS:
      if(msg->p != 0)
      {
        AbsRect& target = *((AbsRect*)msg->p);
        AbsRect r;
        ResolveRect(*self, &r);
        r.left += self->realpadding.left;
        r.top += self->realpadding.top;
        r.right -= self->realpadding.right;
        r.bottom -= self->realpadding.bottom;

        if(msg->subtype == FGSCROLLBAR_SCROLLTO)
        {
          target.left += r.left + self->control.element.padding.left - self->realpadding.left;
          target.top += r.top + self->control.element.padding.top - self->realpadding.top;
          target.right += r.left + self->control.element.padding.left - self->realpadding.left;
          target.bottom += r.top + self->control.element.padding.top - self->realpadding.top;
        }

        float left = bssmax(r.left - target.left, 0);
        float top = bssmax(r.top - target.top, 0);
        float right = bssmin(r.right - target.right, 0);
        float bottom = bssmin(r.bottom - target.bottom, 0);
        float x = (left == 0 && target.right > r.left) ? right : left;
        float y = (top == 0 && target.bottom > r.top) ? bottom : top;
        if(target.right - target.left > r.right - r.left) x = r.left - target.left;
        if(target.bottom - target.top > r.bottom - r.top) y = r.top - target.top;

        fgScrollbar_ApplyPadding(self, x, y);
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_REORDERCHILD:
    if(msg->e != 0 && !(msg->e->flags&FGFLAGS_INTERNAL) && !msg->e2)
    {
      FG_Msg m = *msg;
      m.e2 = &self->bg[0];
      return fgControl_HoverMessage(&self->control, &m);
    }
    break;
  case FG_ADDCHILD:
    if(msg->e != 0 && !(msg->e->flags&FGFLAGS_INTERNAL) && !msg->e2)
    {
      FG_Msg m = *msg;
      m.e2 = &self->bg[0];
      return fgControl_HoverMessage(&self->control, &m);
    }
    break;
  case FG_SETDIM:
    if(fgControl_HoverMessage(&self->control, msg))
    {
      fgScrollbar_Recalc(self);
      return FG_ACCEPT;
    }
    return 0;
  case FG_GETCLASSNAME:
    return (size_t)"Scrollbar";
  }
  return fgControl_HoverMessage(&self->control, msg);
}