// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"
#include "feathercpp.h"

void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgText_Destroy, (FN_MESSAGE)&fgText_Message);

  self->cache = 0;
  if(color) fgChild_IntMessage(*self, FG_SETCOLOR, color, 0);
  if(text) _sendmsg<FG_SETTEXT, void*>(*self, text);
  if(font) _sendmsg<FG_SETFONT, void*>(*self, font);
}

void FG_FASTCALL fgText_Destroy(fgText* self)
{
  assert(self != 0);
  if(self->text != 0) free(self->text);
  if(self->font != 0) fgDestroyFont(self->font);
  fgChild_Destroy(&self->element);
}

size_t FG_FASTCALL fgText_Message(fgText* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgChild_Message(&self->element, msg);
    self->text = 0;
    self->color.color = 0;
    self->font = 0;
    return 1;
  case FG_SETTEXT:
    if(self->text) free(self->text);
    self->text = fgCopyText((const char*)msg->other);
    fgText_Recalc(self);
    fgDirtyElement(&self->element.element);
    return 1;
  case FG_SETFONT:
    switch(msg->subtype)
    {
    case FGTEXT_FONT:
      if(self->font) fgDestroyFont(self->font);
      self->font = 0;
      if(msg->other) self->font = fgCloneFontDPI(msg->other, _sendmsg<FG_GETDPI>(*self));
      break;
    case FGTEXT_LINEHEIGHT:
      self->lineheight = msg->otherf;
      break;
    case FGTEXT_LETTERSPACING:
      self->letterspacing = msg->otherf;
      break;
    }
    fgText_Recalc(self);
    fgDirtyElement(&self->element.element);
    break;
  case FG_SETCOLOR:
    self->color.color = msg->otherint;
    fgDirtyElement(&self->element.element);
    break;
  case FG_GETTEXT:
    return (size_t)self->text;
  case FG_GETFONT:
    switch(msg->subtype)
    {
    case FGTEXT_FONT: return reinterpret_cast<size_t>(self->font);
    case FGTEXT_LINEHEIGHT: return *reinterpret_cast<size_t*>(&self->lineheight);
    case FGTEXT_LETTERSPACING: return *reinterpret_cast<size_t*>(&self->letterspacing);
    }
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
      AbsVec center = ResolveVec(&self->element.element.center, &area);
      self->cache = fgDrawFont(self->font, !self->text ? "" : self->text, self->lineheight, self->letterspacing, self->color.color, &area, self->element.element.rotation, &center, self->element.flags, self->cache);
    }
    break;
  case FG_SETDPI:
    (*self)->SetFont(self->font); // By setting the font to itself we'll clone it into the correct DPI
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgText";
  }
  return fgChild_Message(&self->element, msg);
}

FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&FGCHILD_EXPAND))
  {
    AbsRect area;
    ResolveRect(*self, &area);
    fgFontSize(self->font, !self->text ? "" : self->text, self->lineheight, self->letterspacing, &area, self->element.flags);
    CRect adjust = self->element.element.area;
    if(self->element.flags&FGCHILD_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left;
    if(self->element.flags&FGCHILD_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}