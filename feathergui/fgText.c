// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgText.h"

void FG_FASTCALL fgText_Init(fgText* self, char* text, void* font, unsigned int color, fgFlag flags, fgChild* parent, const fgElement* element)
{
  assert(self != 0);
  fgChild_Init(&self->element, flags, parent, element);
  self->text = 0;
  self->color = 0;
  self->font = 0;
  self->element.destroy = &fgText_Destroy;
  self->element.message = &fgText_Message;

  fgChild_IntMessage((fgChild*)self, FG_SETFONTCOLOR, color, 0);
  fgChild_VoidMessage((fgChild*)self, FG_SETTEXT, text);
  fgChild_VoidMessage((fgChild*)self, FG_SETFONT, font);
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
  case FG_SETFONTCOLOR:
    self->color = msg->otherint;
    break;
  case FG_GETTEXT:
    return (size_t)self->text;
  case FG_GETFONT:
    if(self->font)
      return (size_t)self->font;
    break;
  case FG_GETFONTCOLOR:
    return self->color;
  case FG_MOVE:
    if(!(msg->otheraux & 1) && (msg->otheraux&(2 | 4)))
      fgText_Recalc(self);
    break;
  case FG_DRAW:
    fgDrawFont(self->font, !self->text ? "" : self->text, self->color, (AbsRect*)msg->other, self->element.flags);
    break;
  case FG_GETCLASSNAME:
    return (size_t)"fgText";
  }
  return fgChild_Message(&self->element, msg);
}

FG_EXTERN void FG_FASTCALL fgText_Recalc(fgText* self)
{
  if(self->font && (self->element.flags&(FGCHILD_EXPANDX|FGCHILD_EXPANDY)))
  {
    AbsRect area;
    ResolveRect((fgChild*)self, &area);
    fgFontSize(self->font, !self->text ? "" : self->text, &area, self->element.flags);
    CRect adjust = self->element.element.area;
    if(self->element.flags&FGCHILD_EXPANDX)
    {
      adjust.left.abs = area.left;
      adjust.right.abs = area.right;
    }
    if(self->element.flags&FGCHILD_EXPANDY)
    {
      adjust.top.abs = area.left;
      adjust.bottom.abs = area.right;
    }
    fgChild_VoidMessage((fgChild*)self, FG_SETAREA, &adjust);
  }
}