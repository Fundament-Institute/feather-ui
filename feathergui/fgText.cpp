// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"
#include "feathercpp.h"
#include "bss-util/cDynArray.h"
#include <math.h>

fgElement* FG_FASTCALL fgText_Create(char* text, void* font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  fgElement* r = fgCreate("Text", parent, next, name, flags, transform);
  if(color) fgIntMessage(r, FG_SETCOLOR, color, 0);
  if(text) _sendmsg<FG_SETTEXT, void*>(r, text);
  if(font) _sendmsg<FG_SETFONT, void*>(r, font);
  return r;
}
void FG_FASTCALL fgText_Init(fgText* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  self->cache = 0;
  memset(&self->text, 0, sizeof(fgVectorString));
  memset(&self->buf, 0, sizeof(fgVectorUTF32));
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (fgDestroy)&fgText_Destroy, (fgMessage)&fgText_Message);
}

void FG_FASTCALL fgText_Destroy(fgText* self)
{
  assert(self != 0);
  ((bss_util::cDynArray<int>*)&self->text)->~cDynArray();
  ((bss_util::cDynArray<char>*)&self->buf)->~cDynArray();
  if(self->font != 0) fgDestroyFont(self->font);
  fgElement_Destroy(&self->element);
}

size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(&self->element, msg);
    self->color.color = 0;
    self->font = 0;
    self->lineheight = 0;
    self->letterspacing = 0;
    return FG_ACCEPT;
  case FG_SETTEXT:
    ((bss_util::cDynArray<int>*)&self->text)->Clear();
    ((bss_util::cDynArray<char>*)&self->buf)->Clear();
    if(msg->other)
    {
      ((bss_util::cDynArray<char>*)&self->buf)->operator=(bss_util::cArraySlice<const char>((const char*)msg->other, strlen((const char*)msg->other) + 1));
      size_t len = UTF8toUTF32(self->buf.p, -1, 0, 0);
      ((bss_util::cDynArray<int>*)&self->text)->Reserve(len);
      self->text.l = UTF8toUTF32(self->buf.p, -1, self->text.p, self->text.s);
    }
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    return FG_ACCEPT;
  case FG_SETFONT:
    if(self->font) fgDestroyFont(self->font);
    self->font = 0;
    if(msg->other)
    {
      size_t dpi = _sendmsg<FG_GETDPI>(*self);
      unsigned int fontdpi;
      unsigned int fontsize;
      fgFontGet(msg->other, 0, &fontsize, &fontdpi);
      self->font = (dpi == fontdpi) ? fgCloneFont(msg->other) : fgCopyFont(msg->other, fontsize, fontdpi);
    }
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    return FG_ACCEPT;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->otherf;
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    return FG_ACCEPT;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->otherf;
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    return FG_ACCEPT;
  case FG_SETCOLOR:
    self->color.color = (unsigned int)msg->otherint;
    fgDirtyElement(&self->element.transform);
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
      float scale = (!msg->otheraux || !fgroot_instance->dpi) ? 1.0 : (fgroot_instance->dpi / (float)msg->otheraux);
      area.left *= scale;
      area.top *= scale;
      area.right *= scale;
      area.bottom *= scale;
      AbsVec center = ResolveVec(&self->element.transform.center, &area);
      self->cache = fgDrawFont(self->font, !self->text.p ? &UNICODE_TERMINATOR : self->text.p, self->lineheight, self->letterspacing, self->color.color, &area, self->element.transform.rotation, &center, self->element.flags, self->cache);
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
    if(self->element.flags&FGELEMENT_EXPANDX)
      area.right = area.left;
    if(self->element.flags&FGELEMENT_EXPANDY)
      area.bottom = area.top;

    fgFontSize(self->font, !self->text.p ? &UNICODE_TERMINATOR : self->text.p, self->lineheight, self->letterspacing, &area, self->element.flags);
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left + self->element.padding.left + self->element.padding.right;
    if(self->element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top + self->element.padding.top + self->element.padding.bottom;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}