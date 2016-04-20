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
    if(self->font) fgDestroyFont(self->font);
    self->font = 0;
    if(msg->other) self->font = fgCloneFont(msg->other);
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
    return (size_t)self->font;
  case FG_GETCOLOR:
    return self->color.color;
  case FG_MOVE:
    if(!(msg->otheraux & 1) && (msg->otheraux&(2 | 4)))
      fgText_Recalc(self);
    break;
  case FG_DRAW:
    if(self->font != 0 && !(msg->subtype & 1))
    {
      AbsVec center = ResolveVec(&self->element.element.center, (AbsRect*)msg->other);
      self->cache = fgDrawFont(self->font, !self->text ? "" : self->text, self->color.color, (AbsRect*)msg->other, self->element.element.rotation, &center, self->element.flags, self->cache);
    }
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
    fgFontSize(self->font, !self->text ? "" : self->text, &area, self->element.flags);
    CRect adjust = self->element.element.area;
    if(self->element.flags&FGCHILD_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left;
    if(self->element.flags&FGCHILD_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.bottom - area.top;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}