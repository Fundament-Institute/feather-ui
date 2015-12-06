// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"

void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element)
{
  assert(self != 0);
  memset(self, 0, sizeof(fgText));
  fgChild_Init(&self->element, flags, parent, element);
  self->element.destroy = &fgText_Destroy;
  self->element.message = &fgText_Message;

  if(color) fgChild_IntMessage((fgChild*)self, FG_SETCOLOR, color, 0);
  if(text) fgChild_VoidMessage((fgChild*)self, FG_SETTEXT, text);
  if(font) fgChild_VoidMessage((fgChild*)self, FG_SETFONT, font);
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
  case FG_SETTEXT:
    if(self->text) free(self->text);
    self->text = fgCopyText(msg->other);
    fgText_Recalc(self);
    return 0;
  case FG_SETFONT:
    if(self->font) fgDestroyFont(self->font);
    self->font = 0;
    if(msg->other) self->font = fgCloneFont(msg->other);
    fgText_Recalc(self);
    break;
  case FG_SETCOLOR:
    self->color = msg->otherint;
    break;
  case FG_GETTEXT:
    return (size_t)self->text;
  case FG_GETFONT:
    return (size_t)self->font;
  case FG_GETCOLOR:
    return self->color;
  case FG_MOVE:
    if(!(msg->otheraux & 1) && (msg->otheraux&(2 | 4)))
      fgText_Recalc(self);
    break;
  case FG_DRAW:
  {
    AbsVec center = ResolveVec(&self->element.element.center, (AbsRect*)msg->other);
    fgDrawFont(self->font, !self->text ? "" : self->text, self->color, (AbsRect*)msg->other, self->element.element.rotation, &center, self->element.flags);
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
    ResolveRect((fgChild*)self, &area);
    fgFontSize(self->font, !self->text ? "" : self->text, &area, self->element.flags);
    CRect adjust = self->element.element.area;
    if(self->element.flags&FGCHILD_EXPANDX)
      adjust.right.abs = adjust.left.abs + area.right - area.left;
    if(self->element.flags&FGCHILD_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + area.right - area.left;
    fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &adjust);
  }
}