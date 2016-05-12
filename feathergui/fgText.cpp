// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"
#include "feathercpp.h"

void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform)
{
  self->cache = 0;
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, (FN_DESTROY)&fgText_Destroy, (FN_MESSAGE)&fgText_Message);

  if(color) fgIntMessage(*self, FG_SETCOLOR, color, 0);
  if(text) _sendmsg<FG_SETTEXT, void*>(*self, text);
  if(font) _sendmsg<FG_SETFONT, void*>(*self, font);
}

void FG_FASTCALL fgText_Destroy(fgText* self)
{
  assert(self != 0);
  if(self->text != 0) free(self->text);
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
    self->text = 0;
    self->color.color = 0;
    self->font = 0;
    self->lineheight = 0;
    self->letterspacing = 0;
    return 1;
  case FG_SETTEXT:
    if(self->text) free(self->text);
    self->text = fgCopyText((const char*)msg->other);
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    return 1;
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
    break;
  case FG_SETLINEHEIGHT:
    self->lineheight = msg->otherf;
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    break;
  case FG_SETLETTERSPACING:
    self->letterspacing = msg->otherf;
    fgText_Recalc(self);
    fgDirtyElement(&self->element.transform);
    break;
  case FG_SETCOLOR:
    self->color.color = (unsigned int)msg->otherint;
    fgDirtyElement(&self->element.transform);
    break;
  case FG_GETTEXT:
    return (size_t)self->text;
  case FG_GETFONT:
    return reinterpret_cast<size_t>(self->font);
  case FG_GETLINEHEIGHT:
    return *reinterpret_cast<size_t*>(&self->lineheight);
  case FG_GETLETTERSPACING:
    return *reinterpret_cast<size_t*>(&self->letterspacing);
  case FG_GETCOLOR:
    return self->color.color;
  case FG_MOVE:
    if(!(msg->otheraux & 1) && (msg->otheraux&(2 | 4)))
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
      self->cache = fgDrawFont(self->font, !self->text ? "" : self->text, self->lineheight, self->letterspacing, self->color.color, &area, self->element.transform.rotation, &center, self->element.flags, self->cache);
    }
    break;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgText";
  }
  return fgElement_Message(&self->element, msg);
}

FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&FGELEMENT_EXPAND))
  {
    AbsRect area;
    ResolveRect(*self, &area);
    fgFontSize(self->font, !self->text ? "" : self->text, self->lineheight, self->letterspacing, &area, self->element.flags);
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left;
    if(self->element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}