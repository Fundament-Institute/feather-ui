// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"
#include "feathercpp.h"
#include "bss-util/cDynArray.h"
#include <math.h>

fgElement* fgText_Create(char* text, fgFont font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement* r = fgroot_instance->backend.fgCreate("Text", parent, next, name, flags, transform, units);
  if(color) _sendmsg<FG_SETCOLOR, size_t>(r, color);
  if(text) _sendmsg<FG_SETTEXT, void*>(r, text);
  if(font) _sendmsg<FG_SETFONT, void*>(r, font);
  return r;
}
void fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  static_assert(sizeof(fgVectorUTF8) == sizeof(bss_util::cDynArray<char>), "DynArray size mismatch");
  static_assert(sizeof(fgVectorUTF16) == sizeof(bss_util::cDynArray<wchar_t>), "DynArray size mismatch");
  static_assert(sizeof(fgVectorUTF32) == sizeof(bss_util::cDynArray<int>), "DynArray size mismatch");
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgText_Destroy, (fgMessage)&fgText_Message);
}

void fgText_Destroy(fgText* self)
{
  assert(self != 0);
  if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, self->layout);
  if(self->font != 0) fgroot_instance->backend.fgDestroyFont(self->font);
  self->font = 0;
  fgElement_Destroy(&self->element);
  ((bss_util::cDynArray<int>*)&self->text32)->~cDynArray();
  ((bss_util::cDynArray<wchar_t>*)&self->text16)->~cDynArray();
  ((bss_util::cDynArray<char>*)&self->text8)->~cDynArray();
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
      size_t len = fgUTF16toUTF8(text16->p, text16->l, 0, 0);
      ((bss_util::cDynArray<char>*)text8)->Reserve(len);
      text8->l = fgUTF16toUTF8(text16->p, text16->l, text8->p, text8->s);
    }
    if(text32->l)
    {
      size_t len = fgUTF32toUTF8(text32->p, text32->l, 0, 0);
      ((bss_util::cDynArray<char>*)text8)->Reserve(len);
      text8->l = fgUTF32toUTF8(text32->p, text32->l, text8->p, text8->s);
    }
    return reinterpret_cast<fgVector*>(text8);
  case FGTEXTFMT_UTF16:
    if(text16->l)
      return reinterpret_cast<fgVector*>(text16);
    if(text8->l)
    {
      size_t len = fgUTF8toUTF16(text8->p, text8->l, 0, 0);
      ((bss_util::cDynArray<wchar_t>*)text16)->Reserve(len);
      text16->l = fgUTF8toUTF16(text8->p, text8->l, text16->p, text16->s);
    }
    if(text32->l)
    {
      size_t len = fgUTF32toUTF16(text32->p, text32->l, 0, 0);
      ((bss_util::cDynArray<wchar_t>*)text16)->Reserve(len);
      text16->l = fgUTF32toUTF16(text32->p, text32->l, text16->p, text16->s);
    }
    return reinterpret_cast<fgVector*>(text16);
  case FGTEXTFMT_UTF32:
    if(text32->l)
      return reinterpret_cast<fgVector*>(text32);
    if(text8->l)
    {
      size_t len = fgUTF8toUTF32(text8->p, text8->l, 0, 0);
      ((bss_util::cDynArray<int>*)text32)->Reserve(len);
      text32->l = fgUTF8toUTF32(text8->p, text8->l, text32->p, text32->s);
    }
    if(text16->l)
    {
      size_t len = fgUTF16toUTF32(text16->p, text16->l, 0, 0);
      ((bss_util::cDynArray<int>*)text32)->Reserve(len);
      text32->l = fgUTF16toUTF32(text16->p, text16->l, text32->p, text32->s);
    }
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
    self->layout = 0;
    self->lineheight = 0; // lineheight must be zero'd before a potential transform unit resolution.
    memset(&self->text32, 0, sizeof(fgVectorUTF32));
    memset(&self->text16, 0, sizeof(bss_util::cDynArray<wchar_t>));
    memset(&self->text8, 0, sizeof(fgVectorUTF8));
    fgElement_Message(&self->element, msg);
    self->color.color = 0;
    self->font = 0;
    self->letterspacing = 0;
    return FG_ACCEPT;
  case FG_SETTEXT:
    ((bss_util::cDynArray<int>*)&self->text32)->Clear();
    ((bss_util::cDynArray<wchar_t>*)&self->text16)->Clear();
    ((bss_util::cDynArray<char>*)&self->text8)->Clear();
    if(msg->p)
    {
      switch(msg->subtype)
      {
      case FGTEXTFMT_UTF8:
        ((bss_util::cDynArray<char>*)&self->text8)->operator=(bss_util::cArraySlice<const char>((const char*)msg->p, !msg->u2 ? (strlen((const char*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_UTF16:
        ((bss_util::cDynArray<wchar_t>*)&self->text16)->operator=(bss_util::cArraySlice<const wchar_t>((const wchar_t*)msg->p, !msg->u2 ? (wcslen((const wchar_t*)msg->p) + 1) : msg->u2));
        break;
      case FGTEXTFMT_UTF32:
      {
        int* txt = (int*)msg->p;
        size_t len = msg->u2;
        if(!len)
          while(txt[len++] != 0);
        ((bss_util::cDynArray<int>*)&self->text32)->operator=(bss_util::cArraySlice<const int>((const int*)msg->p, len));
      }
        break;
      }
    }
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETFONT:
  {
    if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, self->layout);
    self->layout = 0;
    void* oldfont = self->font; // We can't delete this up here because it may rely on the same font we're setting.
    self->font = 0;
    if(msg->p)
    {
      assert(msg->p != (void*)0xcdcdcdcdcdcdcdcd);
      fgFontDesc desc;
      fgroot_instance->backend.fgFontGet(msg->p, &desc);
      fgIntVec dpi = self->element.GetDPI();
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
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return self->color.color;
  case FG_MOVE:
    if((msg->subtype != FG_SETAREA || msg->p) && !(msg->u2 & FGMOVE_PROPAGATE) && (msg->u2 & FGMOVE_RESIZE))
      fgText_Recalc(self);
    break;
  case FG_SETAREA:
    fgElement_Message(&self->element, msg);
    fgText_Recalc(self);
    break;
  case FG_DRAW:
    if(self->font != 0 && !(msg->subtype & 1))
    {
      AbsRect area = *(AbsRect*)msg->p;
      fgDrawAuxData* data = (fgDrawAuxData*)msg->p2;
      fgScaleRectDPI(&area, data->dpi.x, data->dpi.y);
      fgSnapAbsRect(area, self->element.flags);
      AbsVec center = ResolveVec(&self->element.transform.center, &area);
      fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
      if(v)
        fgroot_instance->backend.fgDrawFont(self->font, v->p, v->l, self->lineheight, self->letterspacing, self->color.color, &area, self->element.transform.rotation, &center, self->element.flags, data, self->layout);
    }
    break;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_GETCLASSNAME:
    return (size_t)"Text";
  }
  return fgElement_Message(&self->element, msg);
}

void fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&FGELEMENT_EXPAND) && !(self->element.flags&FGELEMENT_SILENT))
  {
    assert(!isnan(self->element.transform.area.left.abs) && !isnan(self->element.transform.area.top.abs) && !isnan(self->element.transform.area.right.abs) && !isnan(self->element.transform.area.bottom.abs));
    AbsRect area;
    ResolveRect(*self, &area);
    if(self->element.flags&FGELEMENT_EXPANDX) // If maxdim is -1, this will translate into a -1 maxdim for the text and properly deal with all resizing cases.
      area.right = area.left + self->element.maxdim.x;
    if(self->element.flags&FGELEMENT_EXPANDY)
      area.bottom = area.top + self->element.maxdim.y;
    fgVector* v = fgText_Conversion(fgroot_instance->backend.BackendTextFormat, &self->text8, &self->text16, &self->text32);
    if(v)
      self->layout = fgroot_instance->backend.fgFontLayout(self->font, v->p, v->l, self->lineheight, self->letterspacing, &area, self->element.flags, self->layout);
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left + self->element.padding.left + self->element.padding.right + self->element.margin.left + self->element.margin.right;
    if(self->element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top + self->element.padding.top + self->element.padding.bottom + self->element.margin.top + self->element.margin.bottom;
    assert(!isnan(adjust.left.abs) && !isnan(adjust.top.abs) && !isnan(adjust.right.abs) && !isnan(adjust.bottom.abs));
    FG_Msg msg = { 0 };
    msg.type = FG_SETAREA;
    msg.p = &adjust;
    fgElement_Message(&self->element, &msg);
    //_sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}