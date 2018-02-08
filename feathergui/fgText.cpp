// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgText.h"
#include "bss-util/DynArray.h"
#include <math.h>

fgElement* fgText_Create(char* text, fgFont font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement* r = fgroot_instance->backend.fgCreate("Text", parent, next, name, flags, transform, units);
  if(color) _sendmsg<FG_SETCOLOR, size_t>(r, color);
  if(text) _sendmsg<FG_SETTEXT, void*>(r, text);
  if(font) _sendmsg<FG_SETFONT, void*>(r, font);
  return r;
}
void fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  static_assert(sizeof(fgVectorUTF8) == sizeof(bss::DynArray<char>), "DynArray size mismatch");
  static_assert(sizeof(fgVectorUTF16) == sizeof(bss::DynArray<wchar_t>), "DynArray size mismatch");
  static_assert(sizeof(fgVectorUTF32) == sizeof(bss::DynArray<int>), "DynArray size mismatch");
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgText_Destroy, (fgMessage)&fgText_Message);
}

void fgText_WipeDynText(fgText* self)
{
  if(!self->text32.s && self->text32.l > 0) // If capacity is zero but the length is nonzero, this is a temporary pointer to an external string
  {
    self->text32.l = 0;
    self->text32.p = 0;
  }
  if(!self->text16.s && self->text16.l > 0)
  {
    self->text16.l = 0;
    self->text16.p = 0;
  }
  if(!self->text8.s && self->text8.l > 0)
  {
    self->text8.l = 0;
    self->text8.p = 0;
  }
}
void fgText_Destroy(fgText* self)
{
  assert(self != 0);
  if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, 0, self->layout);
  if(self->font != 0) fgroot_instance->backend.fgDestroyFont(self->font);
  self->font = 0;
  fgElement_Destroy(&self->element);
  fgText_WipeDynText(self);
  ((bss::DynArray<int>*)&self->text32)->~DynArray();
  ((bss::DynArray<wchar_t>*)&self->text16)->~DynArray();
  ((bss::DynArray<char>*)&self->text8)->~DynArray();
}

inline fgVector* fgText_Conversion(int type, fgVectorUTF8* text8, fgVectorUTF16* text16, fgVectorUTF32* text32)
{
  switch(type)
  {
  case FGTEXTFMT_UTF8:
    if(text8->l)
      return reinterpret_cast<fgVector*>(text8);
    if(text16->l)
    {
      assert(!text16->p[text16->l - 1]);
      size_t len = fgUTF16toUTF8(text16->p, text16->l, 0, 0);
      ((bss::DynArray<char>*)text8)->SetCapacity(len);
      text8->l = fgUTF16toUTF8(text16->p, text16->l, text8->p, text8->s);
    }
    if(text32->l)
    {
      assert(!text32->p[text32->l - 1]);
      size_t len = fgUTF32toUTF8(text32->p, text32->l, 0, 0);
      ((bss::DynArray<char>*)text8)->SetCapacity(len);
      text8->l = fgUTF32toUTF8(text32->p, text32->l, text8->p, text8->s);
    }
    if(!text8->l) // If this is still true, it means everything was empty
      return 0;
    return reinterpret_cast<fgVector*>(text8);
  case FGTEXTFMT_UTF16:
    if(text16->l)
      return reinterpret_cast<fgVector*>(text16);
    if(text8->l)
    {
      assert(!text8->p[text8->l - 1]);
      size_t len = fgUTF8toUTF16(text8->p, text8->l, 0, 0);
      ((bss::DynArray<wchar_t>*)text16)->SetCapacity(len);
      text16->l = fgUTF8toUTF16(text8->p, text8->l, text16->p, text16->s);
    }
    if(text32->l)
    {
      assert(!text32->p[text32->l - 1]);
      size_t len = fgUTF32toUTF16(text32->p, text32->l, 0, 0);
      ((bss::DynArray<wchar_t>*)text16)->SetCapacity(len);
      text16->l = fgUTF32toUTF16(text32->p, text32->l, text16->p, text16->s);
    }
    if(!text16->l)
      return 0;
    return reinterpret_cast<fgVector*>(text16);
  case FGTEXTFMT_UTF32:
    if(text32->l)
      return reinterpret_cast<fgVector*>(text32);
    if(text8->l)
    {
      assert(!text8->p[text8->l - 1]);
      size_t len = fgUTF8toUTF32(text8->p, text8->l, 0, 0);
      ((bss::DynArray<int>*)text32)->SetCapacity(len);
      text32->l = fgUTF8toUTF32(text8->p, text8->l, text32->p, text32->s);
    }
    if(text16->l)
    {
      assert(!text16->p[text16->l - 1]);
      size_t len = fgUTF16toUTF32(text16->p, text16->l, 0, 0);
      ((bss::DynArray<int>*)text32)->SetCapacity(len);
      text32->l = fgUTF16toUTF32(text16->p, text16->l, text32->p, text32->s);
    }
    if(!text32->l)
      return 0;
    return reinterpret_cast<fgVector*>(text32);
  }
  return 0;
}

size_t fgText_Message(fgText* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    bss::memsubset<fgText, fgElement>(self, 0); // lineheight must be zero'd before a potential transform unit resolution.
    fgElement_Message(&self->element, msg);
    return FG_ACCEPT;
  case FG_CLONE:
    if(msg->e)
    {
      fgElement_Message(&self->element, msg);
      fgText* hold = reinterpret_cast<fgText*>(msg->e);
      bss::memsubset<fgText, fgElement>(hold, 0);
      fgText_WipeDynText(self);
      reinterpret_cast<bss::DynArray<int>&>(hold->text32) = reinterpret_cast<bss::DynArray<int>&>(self->text32);
      reinterpret_cast<bss::DynArray<wchar_t>&>(hold->text16) = reinterpret_cast<bss::DynArray<wchar_t>&>(self->text16);
      reinterpret_cast<bss::DynArray<char>&>(hold->text8) = reinterpret_cast<bss::DynArray<char>&>(self->text8);
      if(self->font)
        hold->font = fgroot_instance->backend.fgCloneFont(self->font, 0);
      hold->color = self->color;
      hold->lineheight = self->lineheight;
      hold->letterspacing = self->letterspacing;
      fgText_Recalc(hold);
    }
    return sizeof(fgText);
  case FG_SETTEXT:
    if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, 0, self->layout);
    self->layout = 0;
    fgText_WipeDynText(self);
    ((bss::DynArray<int>*)&self->text32)->Clear();
    ((bss::DynArray<wchar_t>*)&self->text16)->Clear();
    ((bss::DynArray<char>*)&self->text8)->Clear();
    if(msg->p)
    {
      switch(msg->subtype)
      {
      case FGTEXTFMT_UTF8:
        ((bss::DynArray<char>*)&self->text8)->operator=(bss::Slice<const char>((const char*)msg->p, !msg->u2 ? (strlen((const char*)msg->p) + 1) : msg->u2));
        if(((bss::DynArray<char>*)&self->text8)->Back() != 0)
          ((bss::DynArray<char>*)&self->text8)->Add(0);
        break;
      case FGTEXTFMT_DYNAMIC_UTF8:
        self->text8.p = (char*)msg->p;
        self->text8.l = !msg->u2 ? (strlen((const char*)msg->p) + 1) : msg->u2;
        break;
      case FGTEXTFMT_UTF16:
        ((bss::DynArray<wchar_t>*)&self->text16)->operator=(bss::Slice<const wchar_t>((const wchar_t*)msg->p, !msg->u2 ? (wcslen((const wchar_t*)msg->p) + 1) : msg->u2));
        if(((bss::DynArray<wchar_t>*)&self->text16)->Back() != 0)
          ((bss::DynArray<wchar_t>*)&self->text16)->Add(0);
        break;
      case FGTEXTFMT_DYNAMIC_UTF16:
        self->text16.p = (wchar_t*)msg->p;
        self->text16.l = !msg->u2 ? (wcslen((const wchar_t*)msg->p) + 1) : msg->u2;
        break;
      case FGTEXTFMT_UTF32:
      case FGTEXTFMT_DYNAMIC_UTF32:
      {
        int* txt = (int*)msg->p;
        size_t len = msg->u2;
        if(!len)
          while(txt[len++] != 0);
        if(msg->subtype == FGTEXTFMT_DYNAMIC_UTF32)
        {
          self->text32.p = txt;
          self->text32.l = len;
        }
        else
        {
          ((bss::DynArray<int>*)&self->text32)->operator=(bss::Slice<const int>((const int*)msg->p, len));
          if(((bss::DynArray<int>*)&self->text32)->Back() != 0)
            ((bss::DynArray<int>*)&self->text32)->Add(0);
        }
      }
        break;
      }
    }
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETFONT:
  {
    if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, 0, self->layout);
    self->layout = 0;
    void* oldfont = self->font; // We can't delete this up here because it may rely on the same font we're setting.
    self->font = 0;
    if(msg->p)
    {
      assert(msg->p != (void*)0xcdcdcdcdcdcdcdcd);
      fgFontDesc desc;
      fgroot_instance->backend.fgFontGet(msg->p, &desc);
      self->fontlineheight = desc.lineheight;
      AbsVec dpi = self->element.GetDPI();
      bool identical = (dpi.x == desc.dpi.x && dpi.y == desc.dpi.y);
      desc.dpi = dpi;
      self->font = fgroot_instance->backend.fgCloneFont(msg->p, identical ? 0 : &desc);
    }
    if(oldfont) fgroot_instance->backend.fgDestroyFont(oldfont);
    
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
  }
    return FG_ACCEPT;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->f;
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->f;
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETCOLOR:
    self->color.color = (unsigned int)msg->i;
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_GETTEXT:
  {
    fgVector* v = fgText_Conversion(msg->subtype, &self->text8, &self->text16, &self->text32);
    return !v ? 0 : reinterpret_cast<size_t>(v->p);
  }
  case FG_GETFONT:
    return reinterpret_cast<size_t>((void*)self->font);
  case FG_GETLINEHEIGHT:
    return self->lineheight == 0.0f ? *reinterpret_cast<size_t*>(&self->fontlineheight) : *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return self->color.color;
  case FG_MOVE:
    fgElement_Message(&self->element, msg);
    if((msg->subtype != FG_SETAREA || msg->p) && !(msg->u2 & FGMOVE_PROPAGATE) && (msg->u2 & FGMOVE_RESIZE))
      fgText_Recalc(self);
    return FG_ACCEPT;
  case FG_SETAREA:
    fgElement_Message(&self->element, msg);
    fgText_Recalc(self);
    return FG_ACCEPT;
  case FG_DRAW:
    fgElement_Message(&self->element, msg);
    if(self->font != 0 && !(msg->subtype & 1))
    {
      AbsVec center = ResolveVec(&self->element.transform.center, (AbsRect*)msg->p);
      fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
      AbsRect rect;
      __applyrect(rect, *(AbsRect*)msg->p, self->element.padding);
      if(v)
        fgroot_instance->backend.fgDrawFont(self->font, v->p, v->l, self->lineheight, self->letterspacing, self->color.color, &rect, self->element.transform.rotation, &center, self->element.flags, (fgDrawAuxData*)msg->p2, self->layout);
    }
    return FG_ACCEPT;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_CLEAR:
    return self->element.SetText(0);
  case FG_GETCLASSNAME:
    return (size_t)"Text";
  }
  return fgElement_Message(&self->element, msg);
}

void fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&FGELEMENT_EXPAND) && !(self->element.flags&FGELEMENT_SILENT))
  {
    assert(!std::isnan(self->element.transform.area.left.abs) && !std::isnan(self->element.transform.area.top.abs) && !std::isnan(self->element.transform.area.right.abs) && !std::isnan(self->element.transform.area.bottom.abs));
    AbsRect area;
    ResolveInnerRect(*self, &area);
    if(self->element.flags&FGELEMENT_EXPANDX) // If maxdim is -1, this will translate into a -1 maxdim for the text and properly deal with all resizing cases.
      area.right = area.left + self->element.maxdim.x;
    if(self->element.flags&FGELEMENT_EXPANDY)
      area.bottom = area.top + self->element.maxdim.y;
    fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
    if(v)
      self->layout = fgroot_instance->backend.fgFontLayout(self->font, v->p, v->l, self->lineheight, self->letterspacing, &area, self->element.flags, &self->element.GetDPI(), self->layout);
    
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPAND)
    {
      AbsVec dpi = (*self)->GetDPI();
      if(self->element.flags&FGELEMENT_EXPANDX)
        adjust.right.abs = adjust.left.abs + fgSnapAll<ceilf>(area.right - area.left, dpi.x) + self->element.padding.left + self->element.padding.right + self->element.margin.left + self->element.margin.right;
      if(self->element.flags&FGELEMENT_EXPANDY)
        adjust.bottom.abs = adjust.top.abs + fgSnapAll<ceilf>(area.bottom - area.top, dpi.y) + self->element.padding.top + self->element.padding.bottom + self->element.margin.top + self->element.margin.bottom;
    }
    assert(!std::isnan(adjust.left.abs) && !std::isnan(adjust.top.abs) && !std::isnan(adjust.right.abs) && !std::isnan(adjust.bottom.abs));
    FG_Msg msg = { 0 };
    msg.type = FG_SETAREA;
    msg.p = &adjust;
    fgElement_Message(&self->element, &msg);
    //_sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}