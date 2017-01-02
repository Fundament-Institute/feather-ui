// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"
#include "feathercpp.h"
#include "bss-util/cDynArray.h"
#include <math.h>

fgElement* FG_FASTCALL fgText_Create(char* text, void* font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement* r = fgroot_instance->backend.fgCreate("Text", parent, next, name, flags, transform, units);
  if(color) fgIntMessage(r, FG_SETCOLOR, color, 0);
  if(text) _sendmsg<FG_SETTEXT, void*>(r, text);
  if(font) _sendmsg<FG_SETFONT, void*>(r, font);
  return r;
}
void FG_FASTCALL fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgText_Destroy, (fgMessage)&fgText_Message);
}

void FG_FASTCALL fgText_Destroy(fgText* self)
{
  assert(self != 0);
  if(self->layout != 0) fgroot_instance->backend.fgFontLayout(self->font, 0, 0, 0, 0, 0, 0, self->layout);
  if(self->font != 0) fgroot_instance->backend.fgDestroyFont(self->font);
  self->font = 0;
  fgElement_Destroy(&self->element);
  ((bss_util::cDynArray<int>*)&self->text)->~cDynArray();
  ((bss_util::cDynArray<char>*)&self->buf)->~cDynArray();
}

size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    self->layout = 0;
    self->lineheight = 0; // lineheight must be zero'd before a potential transform unit resolution.
    memset(&self->text, 0, sizeof(fgVectorString));
    memset(&self->buf, 0, sizeof(fgVectorUTF32));
    fgElement_Message(&self->element, msg);
    self->color.color = 0;
    self->font = 0;
    self->letterspacing = 0;
    return FG_ACCEPT;
  case FG_SETTEXT:
    ((bss_util::cDynArray<int>*)&self->text)->Clear();
    ((bss_util::cDynArray<char>*)&self->buf)->Clear();
    if(msg->other)
    {
      if(msg->subtype == FGSETTEXT_UTF8)
      {
        ((bss_util::cDynArray<char>*)&self->buf)->operator=(bss_util::cArraySlice<const char>((const char*)msg->other, strlen((const char*)msg->other) + 1));
        size_t len = fgUTF8toUTF32(self->buf.p, -1, 0, 0);
        ((bss_util::cDynArray<int>*)&self->text)->Reserve(len);
        self->text.l = fgUTF8toUTF32(self->buf.p, -1, self->text.p, self->text.s);
      }
      else if(msg->subtype == FGSETTEXT_UTF32)
      {
        int* txt = (int*)msg->other;
        size_t len = 0;
        while(txt[len++] != 0);
        ((bss_util::cDynArray<int>*)&self->text)->Reserve(len);
        MEMCPY(self->text.p, self->text.s, txt, sizeof(int)*len);
      }
    }


    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETFONT:
    if(self->font) fgroot_instance->backend.fgDestroyFont(self->font);
    self->font = 0;
    if(msg->other)
    {
      assert(msg->other != (void*)0xcdcdcdcdcdcdcdcd);
      fgFontDesc desc;
      fgroot_instance->backend.fgFontGet(msg->other, &desc);
      fgIntVec dpi = self->element.GetDPI();
      bool identical = (dpi.x == desc.dpi.x && dpi.y == desc.dpi.y);
      desc.dpi = dpi;
      self->font = fgroot_instance->backend.fgCloneFont(msg->other, identical ? 0 : &desc);
    }
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->otherf;
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->otherf;
    fgText_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETCOLOR:
    self->color.color = (unsigned int)msg->otherint;
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_GETTEXT:
    return reinterpret_cast<size_t>(self->text.p);
  case FG_GETFONT:
    return reinterpret_cast<size_t>(self->font);
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return self->color.color;
  case FG_MOVE:
    if(!(msg->otheraux & FGMOVE_PROPAGATE) && (msg->otheraux & FGMOVE_RESIZE))
      fgText_Recalc(self);
    break;
  case FG_DRAW:
    if(self->font != 0 && !(msg->subtype & 1))
    {
      AbsRect area = *(AbsRect*)msg->other;
      fgDrawAuxData* data = (fgDrawAuxData*)msg->other2;
      AbsVec scale = { (!data->dpi.x || !fgroot_instance->dpi.x) ? 1.0f : (fgroot_instance->dpi.x / (float)data->dpi.x), (!data->dpi.y || !fgroot_instance->dpi.y) ? 1.0f : (fgroot_instance->dpi.y / (float)data->dpi.y) };
      area.left *= scale.x;
      area.top *= scale.y;
      area.right *= scale.x;
      area.bottom *= scale.y;
      AbsVec center = ResolveVec(&self->element.transform.center, &area);
      fgroot_instance->backend.fgDrawFont(self->font, self->text.p, self->text.l, self->lineheight, self->letterspacing, self->color.color, &area, self->element.transform.rotation, &center, self->element.flags, data, self->layout);
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

void FG_FASTCALL fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&FGELEMENT_EXPAND))
  {
    assert(!isnan(self->element.transform.area.left.abs) && !isnan(self->element.transform.area.top.abs) && !isnan(self->element.transform.area.right.abs) && !isnan(self->element.transform.area.bottom.abs));
    AbsRect area;
    ResolveRect(*self, &area);
    if(self->element.flags&FGELEMENT_EXPANDX) // If maxdim is -1, this will translate into a -1 maxdim for the text and properly deal with all resizing cases.
      area.right = area.left + self->element.maxdim.x;
    if(self->element.flags&FGELEMENT_EXPANDY)
      area.bottom = area.top + self->element.maxdim.y;
    self->layout = fgroot_instance->backend.fgFontLayout(self->font, self->text.p, self->text.l, self->lineheight, self->letterspacing, &area, self->element.flags, self->layout);
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left + self->element.padding.left + self->element.padding.right + self->element.margin.left + self->element.margin.right;
    if(self->element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top + self->element.padding.top + self->element.padding.bottom + self->element.margin.top + self->element.margin.bottom;
    assert(!isnan(adjust.left.abs) && !isnan(adjust.top.abs) && !isnan(adjust.right.abs) && !isnan(adjust.bottom.abs));
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}