// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgResource.h"
#include <stdio.h>
#include "feathercpp.h"

void FG_FASTCALL fgResource_Init(fgResource* self, void* res, const CRect* uv, unsigned int color, fgFlag flags, fgChild* BSS_RESTRICT parent, fgChild* BSS_RESTRICT prev, const fgElement* element)
{
  fgChild_InternalSetup(*self, flags, parent, prev, element, (FN_DESTROY)&fgResource_Destroy, (FN_MESSAGE)&fgResource_Message);
  if(color) fgChild_IntMessage(*self, FG_SETCOLOR, color, 0);
  if(uv) fgSendMsg<FG_SETUV, void*>(*self, (void*)uv);
  if(res) fgSendMsg<FG_SETRESOURCE, void*>(*self, res);
}
void FG_FASTCALL fgResource_Destroy(fgResource* self)
{
  if(self->res) fgDestroyResource(self->res);
  fgChild_Destroy(&self->element);
}
size_t FG_FASTCALL fgResource_Message(fgResource* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgChild_Message(&self->element, msg);
    memset(&self->uv, 0, sizeof(CRect));
    self->uv.right.rel = 1.0f;
    self->uv.bottom.rel = 1.0f;
    self->color.color = 0;
    self->edge.color = 0;
    self->res = 0;
    self->outline = 0;
    return 1;
  case FG_SETUV:
    if(msg->other)
      self->uv = *((CRect*)msg->other);
    fgResource_Recalc(self);
    return 1;
  case FG_SETRESOURCE:
    if(self->res) fgDestroyResource(self->res);
    self->res = 0;
    if(msg->other) self->res = fgCloneResource(msg->other);
    fgResource_Recalc(self);
    break;
  case FG_SETCOLOR:
    if(msg->otheraux != 0)
      self->edge.color = msg->otherint;
    else
      self->color.color = msg->otherint;
    break;
  case FG_SETOUTLINE:
    self->outline = msg->otherf;
    break;
  case FG_GETUV:
    return (size_t)(&self->uv);
  case FG_GETRESOURCE:
    return (size_t)self->res;
  case FG_GETCOLOR:
    if(msg->otherint != 0)
      return self->edge.color;
    else
      return self->color.color;
  case FG_GETOUTLINE:
    return *reinterpret_cast<size_t*>(&self->outline);
  case FG_MOVE:
    if(!(msg->otheraux & 1) && (msg->otheraux&(2 | 4)))
      fgResource_Recalc(self);
    break;
  case FG_DRAW:
  {
    AbsVec center = ResolveVec(&self->element.element.center, (AbsRect*)msg->other);
    if(self->element.flags&FGRESOURCE_LINE)
    {
      AbsRect area = *(AbsRect*)msg->other;
      area.bottom -= 1;
      area.right -= 1;

      // TODO: correctly center and rotate the resulting line.
      fgDrawLine(area.topleft, area.bottomright, self->color.color);
    }
    else
      fgDrawResource(self->res, &self->uv, self->color.color, self->edge.color, self->outline, (AbsRect*)msg->other, self->element.element.rotation, &center, self->element.flags);
  }
  break;
  case FG_GETCLASSNAME:
    return (size_t)"fgResource";
  }
  return fgChild_Message(&self->element, msg);
}

void* FG_FASTCALL fgCreateResourceFile(fgFlag flags, const char* file)
{
  FILE* f = fopen(file, "rb");
  if(!f) return 0;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc(len);
  fread(buf, 1, len, f);
  fclose(f);
  void* r = fgCreateResource(flags, buf, len);
  free(buf);
  return r;
}

void FG_FASTCALL fgResource_Recalc(fgResource* self)
{
  if(self->res && (self->element.flags&FGCHILD_EXPAND))
  {
    AbsVec dim;
    fgResourceSize(self->res, &self->uv, &dim, self->element.flags);
    CRect adjust = self->element.element.area;
    if(self->element.flags&FGCHILD_EXPANDX)
      adjust.right.abs = adjust.left.abs + dim.x;
    if(self->element.flags&FGCHILD_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + dim.y;
    fgSendMsg<FG_SETAREA, void*>(*self, &adjust);
  }
}