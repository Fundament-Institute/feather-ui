// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgWorkspace.h"

void fgWorkspace_Init(fgWorkspace* BSS_RESTRICT self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, fgMsgType units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgWorkspace_Destroy, (fgMessage)&fgWorkspace_Message);
}
void fgWorkspace_Destroy(fgWorkspace* self)
{
  self->scroll->message = (fgMessage)fgScrollbar_Message;
  fgScrollbar_Destroy(&self->scroll);
}
void fgWorkspace_DrawGrid(fgElement* e, const AbsRect* area, const fgDrawAuxData* data, fgElement*)
{
  fgWorkspace* self = reinterpret_cast<fgWorkspace*>(e);
  if(self->gridcolor[0].a > 0) // x-axis lines (going down the y-axis)
  {
    // TODO
  }
  if(self->gridcolor[1].a > 0) // y-axis lines (going across the x-axis)
  {
    // TODO
  }
}
size_t fgWorkspace_Message(fgWorkspace* self, const FG_Msg* msg)
{
  assert(self != 0);
  fgFlag otherint = (fgFlag)msg->u;

  switch(msg->type)
  {
    case FG_CONSTRUCT:
      bss::memsubset<fgWorkspace, fgScrollbar>(self, 0);
      fgScrollbar_Message(&self->scroll, msg);
      fgElement_Init(&self->rulers[0], *self, 0, "Workspace$LeftRuler", FGELEMENT_BACKGROUND | FGWORKSPACE_RULERX | FGFLAGS_INTERNAL, 0, 0);
      fgElement_Init(&self->rulers[1], *self, 0, "Workspace$TopRuler", FGELEMENT_BACKGROUND | FGWORKSPACE_RULERY | FGFLAGS_INTERNAL, 0, 0);
      fgElement_Init(&self->cursors[0], &self->rulers[0], 0, "Workspace$LeftCursor", FGELEMENT_BACKGROUND | FGELEMENT_EXPAND | FGFLAGS_INTERNAL, 0, 0);
      fgElement_Init(&self->cursors[1], &self->rulers[1], 0, "Workspace$TopCursor", FGELEMENT_BACKGROUND | FGELEMENT_EXPAND | FGFLAGS_INTERNAL, 0, 0);
      self->rulers[0].message = (fgMessage)&fgWorkspace_RulerMessage;
      self->rulers[1].message = (fgMessage)&fgWorkspace_RulerMessage;
      return FG_ACCEPT;
    case FG_CLONE:
      if(msg->e)
      {
        fgWorkspace* hold = reinterpret_cast<fgWorkspace*>(msg->e);
        bss::memsubcpy<fgWorkspace, fgScrollbar>(hold, self);
        self->rulers[0].Clone(&hold->rulers[0]);
        self->rulers[1].Clone(&hold->rulers[1]);
        self->cursors[0].Clone(&hold->cursors[0]);
        self->cursors[1].Clone(&hold->cursors[1]);
        fgScrollbar_Message(&self->scroll, msg);
      }
      return sizeof(fgWorkspace);
    case FG_SETFLAG: // If 0 is sent in, disable the flag, otherwise enable. Our internal flag is 1 if clipping disabled, 0 otherwise.
      otherint = bss::bssSetBit<fgFlag>(self->scroll->flags, otherint, msg->u2 != 0);
    case FG_SETFLAGS:
      if(((self->scroll->flags ^ otherint) & FGWORKSPACE_RULERX) != 0)
      {
        // TODO
      }
      break;
    case FG_SETDIM:
      if((msg->subtype&FGDIM_MASK) == FGDIM_FIXED)
      {
        self->gridsize.x = msg->f;
        self->gridsize.y = msg->f2;
        return FG_ACCEPT;
      }
      break;
    case FG_GETDIM:
      if((msg->subtype&FGDIM_MASK) == FGDIM_FIXED)
        return *reinterpret_cast<size_t*>(&self->gridsize);
      break;
    case FG_DRAW:
      fgStandardDraw(*self, (const AbsRect*)msg->p, (const fgDrawAuxData*)msg->p2, msg->subtype & 1, &fgWorkspace_DrawGrid);
      if(self->crosshaircolor.a > 0) // Draw crosshairs over everything, if they are enabled
      {
        // TODO
      }
      return FG_ACCEPT;
    case FG_GETCLASSNAME:
      return (size_t)"Workspace";
  }

  return fgScrollbar_Message(&self->scroll, msg);
}

void fgWorkspace_DrawRuler(fgElement* e, const AbsRect* area, const fgDrawAuxData* data, fgElement*)
{
  // three possibilities: 1/3 height tick, 1/2 height tick, or 2/3 height tick. smallest tick on every grid line, medium every 5 grid lines, large every 100 grid lines.
  if(e->flags&FGWORKSPACE_RULERX)
  {
    // TODO
  }
  else if(e->flags&FGWORKSPACE_RULERY)
  {
    // TODO
  }
}

size_t fgWorkspace_RulerMessage(fgElement* self, const FG_Msg* msg)
{
  switch(msg->type)
  {
  case FG_DRAW:
    fgStandardDraw(self, (const AbsRect*)msg->p, (const fgDrawAuxData*)msg->p2, msg->subtype & 1, &fgWorkspace_DrawRuler);
    return FG_ACCEPT;
  }
  return fgElement_Message(self, msg);
}