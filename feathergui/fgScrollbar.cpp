// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgScrollbar.h"
#include "bss-util\bss_util.h"
#include "feathercpp.h"

void FG_FASTCALL fgScrollbar_Init(fgScrollbar* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgScrollbar_Destroy, (fgMessage)&fgScrollbar_Message);
}
void FG_FASTCALL fgScrollbar_Destroy(fgScrollbar* self)
{
  fgControl_Destroy(&self->control);
}

size_t FG_FASTCALL fgScrollbar_cornerMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
  case FG_MOUSEUP:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
  case FG_MOUSELEAVE:
  case FG_MOUSEMOVE:
    return FG_ACCEPT;
  }
  return fgElement_Message(self, msg);
}

size_t FG_FASTCALL fgScrollbar_bgMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(self, msg);
    _sendsubmsg<FG_ACTION>(self->parent, FGSCROLLBAR_BARCACHE);
    return FG_ACCEPT;
  case FG_ACTION:
    if(self->parent != 0)
      return fgPassMessage(self->parent, msg);
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
  case FG_MOUSEUP:
  case FG_MOUSEDBLCLICK:
  case FG_MOUSEON:
  case FG_MOUSEOFF:
  case FG_MOUSESCROLL:
  case FG_MOUSELEAVE:
  case FG_MOUSEMOVE:
    return FG_ACCEPT;
  case FG_SETAREA:
    size_t ret = fgElement_Message(self, msg);
    if(ret != 0 && self->parent != 0)
      _sendsubmsg<FG_ACTION>(self->parent, FGSCROLLBAR_BARCACHE);
    return ret;
  }

  return fgElement_Message(self, msg);
}

size_t FG_FASTCALL fgScrollbar_barMessage(_FG_SCROLLBAR_INNER* self, const FG_Msg* msg)
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
    self->lastmouse.x = msg->x;
    self->lastmouse.y = msg->y;
    if(self->button->parent != 0)
      _sendsubmsg<FG_ACTION, ptrdiff_t>(self->button->parent, FGSCROLLBAR_BARINIT, self->button->userid);
    break;
  case FG_MOUSEMOVE:
    if(msg->allbtn&FG_MOUSELBUTTON)
    {
      if(self->button->parent != 0)
        _sendsubmsg<FG_ACTION, ptrdiff_t, float>(
          self->button->parent,
          FGSCROLLBAR_BAR,
          self->button->userid,
          (!self->button->userid) ? (self->lastmouse.x - msg->x) : (self->lastmouse.y - msg->y));
    }
    break;
  }
  return fgButton_Message(&self->button, msg);
}

size_t FG_FASTCALL fgScrollbar_buttonMessage(fgButton* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_MOUSEDOWN:
    if(self->control.element.parent != 0)
      _sendsubmsg<FG_ACTION, ptrdiff_t>(self->control.element.parent, FGSCROLLBAR_BUTTON, self->control.element.userid);
    break;
  }
  return fgButton_Message(self, msg);
}

void FG_FASTCALL fgScrollbar_ApplyPadding(fgScrollbar* self, float x, float y);

void FG_FASTCALL fgScrollbar_Recalc(fgScrollbar* self)
{
  AbsRect r;
  ResolveRect(*self, &r);
  AbsVec dim = { r.right - r.left, r.bottom - r.top };

  bool hideh = !!(self->control.element.flags&FGSCROLLBAR_HIDEH);
  bool hidev = !!(self->control.element.flags&FGSCROLLBAR_HIDEV);

  // We have to figure out which scrollbars are visible based on flags and our dimensions
  bool scrollx = !hideh && ((dim.x < self->realsize.x + self->realpadding.left + self->realpadding.right + bssmax(self->barcache.y, 0)) || self->control.element.flags&FGSCROLLBAR_SHOWH);
  bool scrolly = !hidev && ((dim.y < self->realsize.y + self->realpadding.top + self->realpadding.bottom + bssmax(self->barcache.x, 0)) || self->control.element.flags&FGSCROLLBAR_SHOWV);

  // If we are adding or removing a scrollbar, this will change the padding and could change everything else (e.g. for textboxes)
  if((self->barcache.x != 0.0f && scrollx ^ (self->barcache.x > 0)) || (self->barcache.y != 0.0f && scrolly ^ (self->barcache.y > 0)))
  {
    if(scrollx ^ (self->barcache.x > 0)) self->barcache.x = -self->barcache.x;
    if(scrolly ^ (self->barcache.y > 0)) self->barcache.y = -self->barcache.y;
    return fgScrollbar_ApplyPadding(self, 0, 0);
  }

  // If we get this far, the padding has already been taken care of - append or remove scrollbars.
  self->bg[0].SetFlag(FGELEMENT_HIDDEN, self->barcache.x <= 0);
  self->bg[1].SetFlag(FGELEMENT_HIDDEN, self->barcache.y <= 0);

  // Expand area in response to adding scrollbars if necessary
  CRect area = self->control.element.transform.area;
  if(self->control.element.flags & FGELEMENT_EXPANDX)
    area.right.abs = area.left.abs + self->realsize.x + self->realpadding.left + self->realpadding.right + bssmax(self->barcache.y, 0);
  if(self->control.element.flags & FGELEMENT_EXPANDY)
    area.bottom.abs = area.top.abs + self->realsize.y + self->realpadding.top + self->realpadding.bottom + bssmax(self->barcache.x, 0);
  _sendmsg<FG_SETAREA, void*>(*self, &area); // SETAREA will set MAXDIM appropriately

  // If both scrollbars are active, set the margins to exclude them from the corner
  bool exclude = (self->barcache.x >= 0 && self->barcache.y >= 0);
  self->bg[0].SetMargin(AbsRect { 0,0,exclude ? self->barcache.x : 0,0 });
  self->bg[1].SetMargin(AbsRect { 0,0,0,exclude ? self->barcache.y : 0 });
  self->bg[2].SetFlag(FGELEMENT_HIDDEN|FGELEMENT_IGNORE, !exclude);

  // Set bar dimensions appropriately based on padding
  if(self->barcache.x >= 0)
  {
    float d = r.right - r.left - self->realpadding.left - self->realpadding.right - bssmax(self->barcache.y, 0);
    float length = (self->realsize.x <= 0.0f) ? 1.0f : (d / self->realsize.x);
    float pos = (d - self->realsize.x) == 0.0f ? 0.0f : self->control.element.padding.left / (d - self->realsize.x);
    self->bar[0].button->SetArea(CRect { 0, (1.0f - length)*pos, 0, 0, 0, (1.0f - length)*pos + length, 0, 1.0f });
    self->btn[0]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos <= 0.0f);
    self->btn[2]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos >= 1.0f);
    self->bar[0].button->SetFlag(FGCONTROL_DISABLE, length >= 1.0f);
  }
  if(self->barcache.y >= 0)
  {
    float d = r.bottom - r.top - self->realpadding.top - self->realpadding.bottom - bssmax(self->barcache.x, 0);
    float length = (self->realsize.y <= 0.0f) ? 1.0f : (d / self->realsize.y);
    float pos = (d - self->realsize.y) == 0.0f ? 0.0f : self->control.element.padding.top / (d - self->realsize.y);
    self->bar[1].button->SetArea(CRect { 0, 0, 0, (1.0f - length)*pos, 0, 1.0f, 0, (1.0f - length)*pos + length });
    self->btn[1]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos <= 0.0f);
    self->btn[3]->SetFlag(FGCONTROL_DISABLE, length >= 1.0f || pos >= 1.0f);
    self->bar[1].button->SetFlag(FGCONTROL_DISABLE, length >= 1.0f);
  }
}

void FG_FASTCALL fgScrollbar_ApplyPadding(fgScrollbar* self, float x, float y)
{
  self->control.element.padding.left += x;
  self->control.element.padding.top += y;
  if(self->control.element.padding.left > self->realpadding.left || (self->control->flags & FGSCROLLBAR_HIDEH))
    self->control.element.padding.left = self->realpadding.left; // if HIDEH is true we don't let you scroll at all
  if(self->control.element.padding.top > self->realpadding.top || (self->control->flags & FGSCROLLBAR_HIDEV))
    self->control.element.padding.top = self->realpadding.top; // if HIDEV is true we don't let you scroll at all

  AbsRect r;
  ResolveRect(*self, &r);
  float minpaddingx = (r.right - r.left - self->realpadding.left - self->realpadding.right) - self->realsize.x - bssmax(self->barcache.y, 0);
  minpaddingx = bssmin(minpaddingx, self->realpadding.left);
  float minpaddingy = (r.bottom - r.top - self->realpadding.top - self->realpadding.bottom) - self->realsize.y - bssmax(self->barcache.x, 0);
  minpaddingy = bssmin(minpaddingy, self->realpadding.top);
  if(self->control.element.padding.left < minpaddingx)
    self->control.element.padding.left = minpaddingx;
  if(self->control.element.padding.top < minpaddingy)
    self->control.element.padding.top = minpaddingy;

  self->control.element.padding.right = self->realpadding.right + bssmax(self->barcache.y, 0);
  self->control.element.padding.bottom = self->realpadding.bottom + bssmax(self->barcache.x, 0);

  fgScrollbar_Recalc(self); // recalculate scrollbars from our new padding value
}

void FG_FASTCALL fgScrollbar_SetBarcache(fgScrollbar* self)
{
  AbsRect r;
  ResolveRect(&self->bg[0], &r);
  self->barcache.x = r.bottom - r.top;
  ResolveRect(&self->bg[1], &r);
  self->barcache.y = r.right - r.left;
  self->bg[2].SetArea(CRect { -self->barcache.y, 1, -self->barcache.x, 1, 0, 1, 0, 1});
  fgScrollbar_ApplyPadding(self, 0, 0);
}

size_t FG_FASTCALL fgScrollbar_Message(fgScrollbar* self, const FG_Msg* msg)
{
  static const float DEFAULT_LINEHEIGHT = 30.0f;
  assert(self != 0 && msg != 0);

  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgControl_Message(&self->control, msg);
    self->maxdim.x = -1.0f;
    self->maxdim.y = -1.0f;
    memset(&self->realpadding, 0, sizeof(AbsRect));
    memset(&self->barcache, 0, sizeof(AbsVec));
    memset(&self->realsize, 0, sizeof(AbsVec));
    fgElement_Init(&self->bg[0], *self, 0, "Scrollbar:horzbg", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN, &fgTransform_EMPTY);
    fgElement_Init(&self->bg[1], *self, 0, "Scrollbar:vertbg", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN, &fgTransform_EMPTY);
    fgElement_Init(&self->bg[2], *self, 0, "Scrollbar:corner", FGELEMENT_BACKGROUND | FGELEMENT_HIDDEN, &fgTransform_EMPTY);
    fgButton_Init(&self->btn[0], &self->bg[0], 0, "Scrollbar:scrollleft", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgButton_Init(&self->btn[1], &self->bg[1], 0, "Scrollbar:scrolltop", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgButton_Init(&self->btn[2], &self->bg[0], 0, "Scrollbar:scrollright", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgButton_Init(&self->btn[3], &self->bg[1], 0, "Scrollbar:scrollbottom", FGELEMENT_BACKGROUND, &fgTransform_EMPTY);
    fgButton_Init(&self->bar[0].button, &self->bg[0], 0, "Scrollbar:scrollhorz", 0, &fgTransform_EMPTY);
    fgButton_Init(&self->bar[1].button, &self->bg[1], 0, "Scrollbar:scrollvert", 0, &fgTransform_EMPTY);
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
    fgScrollbar_SetBarcache(self);
    return FG_ACCEPT;
  case FG_SETAREA: // Responding to dimension changes must be done in FG_MOVE
    if(self->maxdim.x < 0 && self->maxdim.y < 0) // If this is true maxdim is not being used at all, so pass it through
      break;
    if(msg->other != nullptr)
    {
      AbsRect resolve;
      CRect area = self->control.element.transform.area;
      self->control.element.transform.area = *(CRect*)msg->other;
      ResolveRect(*self, &resolve);

      if((self->maxdim.x < 0 || resolve.right - resolve.left <= self->maxdim.x) && (self->maxdim.y < 0 || resolve.bottom - resolve.top <= self->maxdim.y))
        break; // If the area doesn't need any adjustment, just pass it through

      CRect narea = self->control.element.transform.area;
      self->control.element.transform.area = area;

      if(self->maxdim.x >= 0)
        narea.right.abs = narea.left.abs + std::min<float>(resolve.right - resolve.left, self->maxdim.x);
      if(self->maxdim.y >= 0)
        narea.bottom.abs = narea.top.abs + std::min<float>(resolve.bottom - resolve.top, self->maxdim.y);

      FG_Msg m = *msg;
      m.other = &narea;
      return fgControl_HoverMessage(&self->control, &m);
    }
    return 0;
  case FG_SETPADDING:
    if(msg->other != nullptr)
    {
      AbsRect* padding = (AbsRect*)msg->other;
      char diff = CompareMargins(&self->realpadding, padding);
      memcpy(&self->realpadding, padding, sizeof(AbsRect));

      if(diff) // Only send a move message if the padding change actually changed something
        fgSubMessage(*self, FG_MOVE, FG_SETPADDING, 0, FGMOVE_PADDING);

      fgScrollbar_ApplyPadding(self, 0, 0); // Recalculate scrollbar positions
      return FG_ACCEPT;
    }
    return 0;
  case FG_MOUSESCROLL:
  {
    FG_Msg m { 0 };
    m.type = FG_ACTION;
    m.subtype = FGSCROLLBAR_CHANGE;
    float lineheight = self->control.element.GetLineHeight();
    m.otherfaux = (msg->scrolldelta / 120.0f) * 3.0f * lineheight;
    m.otherf = (msg->scrollhdelta / -120.0f) * lineheight;

    fgPassMessage(*self, &m);
    return FG_ACCEPT;
  }
  case FG_LAYOUTCHANGE:
    if((msg->subtype == FGELEMENT_LAYOUTADD || msg->subtype == FGELEMENT_LAYOUTREORDER) && self->control->last != &self->bg[2])
    {
      self->bg[0].SetParent(*self, 0);
      self->bg[1].SetParent(*self, 0);
      self->bg[2].SetParent(*self, 0);
    }
    {
      fgFlag flags = self->control.element.flags;
      auto area = self->control.element.transform.area;
      area.right.abs = area.left.abs + self->realsize.x;
      area.bottom.abs = area.top.abs + self->realsize.y;
      self->control.element.flags |= FGELEMENT_EXPAND;
      
      size_t dim = _sendmsg<FG_LAYOUTFUNCTION, const void*, void*>(*self, msg, &area);
      self->control.element.flags = flags;
      self->realsize = { area.right.abs - area.left.abs, area.bottom.abs - area.top.abs }; // retrieve real area and then reset to the area of the window.
      if(dim)
      {
        if(flags & FGELEMENT_EXPAND)
          area = self->control.element.transform.area;
        if(flags & FGELEMENT_EXPANDX)
          area.right.abs = area.left.abs + self->realsize.x + self->realpadding.left + self->realpadding.right + bssmax(self->barcache.y, 0);
        if(flags & FGELEMENT_EXPANDY)
          area.bottom.abs = area.top.abs + self->realsize.y + self->realpadding.top + self->realpadding.bottom + bssmax(self->barcache.x, 0);
        if(flags & FGELEMENT_EXPAND)
          self->control.element.SetArea(area);
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
      if(!msg->otherint) self->lastpadding.x = self->control.element.padding.left;
      else self->lastpadding.y = self->control.element.padding.top;
      return FG_ACCEPT;
    case FGSCROLLBAR_BAR:
    {
      AbsRect r;
      ResolveRect(&self->bg[msg->otherint != 0], &r);
      if(!msg->otherint)
        fgScrollbar_ApplyPadding(self, self->lastpadding.x + msg->otherfaux * self->realsize.x / (r.right - r.left - self->realpadding.left - self->realpadding.right - self->bg[0].padding.left - self->bg[0].padding.right) - self->control.element.padding.left, 0);
      else
        fgScrollbar_ApplyPadding(self, 0, self->lastpadding.y + msg->otherfaux * self->realsize.y / (r.bottom - r.top - self->realpadding.top - self->realpadding.bottom - self->bg[1].padding.top - self->bg[1].padding.bottom) - self->control.element.padding.top);
    }
      return FG_ACCEPT;
    case FGSCROLLBAR_PAGE: // By default a page scroll (clicking on the area outside of the bar or hitting pageup/pagedown) attempts to scroll by the width of the container in that direction.
    {
      AbsRect r;
      ResolveRect(*self, &r);
      if(msg->otherint&1)
        fgScrollbar_ApplyPadding(self, 0, (r.bottom - r.top)*(msg->otherint == 1 ? 1 : -1));
      else
        fgScrollbar_ApplyPadding(self, (r.right - r.left)*(msg->otherint == 0 ? 1 : -1), 0);
    }
      return FG_ACCEPT;
    case FGSCROLLBAR_BUTTON:
    {
      float x = msg->otherint < 2 ? self->control.element.GetLineHeight() : -self->control.element.GetLineHeight();
      if(msg->otherint % 2)
        fgScrollbar_ApplyPadding(self, 0, x);
      else
        fgScrollbar_ApplyPadding(self, x, 0);
      return FG_ACCEPT;
    }
    case FGSCROLLBAR_CHANGE: // corresponds to an actual change amount. First argument is x axis, second argument is y axis.
      fgScrollbar_ApplyPadding(self, msg->otherf, msg->otherfaux);
      return FG_ACCEPT;
    case FGSCROLLBAR_BARCACHE:
      fgScrollbar_SetBarcache(self);
      return FG_ACCEPT;
    case FGSCROLLBAR_SCROLLTO:
      if(msg->other != 0)
      {
        AbsRect& target = *((AbsRect*)msg->other);
        AbsRect r;
        ResolveRect(*self, &r);
        float left = self->control.element.padding.left - self->realpadding.left;
        float top = self->control.element.padding.top - self->realpadding.top;
        float right = r.right - r.left - bssmax(self->barcache.y, 0) - self->realpadding.right - self->realpadding.left - self->control.element.padding.left;
        float bottom = r.bottom - r.top - bssmax(self->barcache.x, 0) - self->realpadding.bottom - self->realpadding.left - self->control.element.padding.top;
        left = bssmax(- target.left - left, 0);
        top = bssmax(- target.top - top, 0);
        right = bssmin(right - target.right, 0);
        bottom = bssmin(bottom - target.bottom, 0);
        float x = (left == 0) ? right : left;
        float y = (top == 0) ? bottom : top;
        
        fgScrollbar_ApplyPadding(self, x, y);
      }
      return FG_ACCEPT;
    }
    return 0;
  case FG_SETDIM:
    if(msg->subtype != FGDIM_MAX)
      break;
    self->maxdim.x = msg->otherf;
    self->maxdim.y = msg->otherfaux;
    return FG_ACCEPT;
  case FG_GETDIM:
    if(msg->subtype == FGDIM_MAX)
      return *reinterpret_cast<size_t*>(&self->maxdim);
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Scrollbar";
  }
  return fgControl_HoverMessage(&self->control, msg);
}