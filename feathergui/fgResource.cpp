// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgResource.h"
#include <stdio.h>
#include "feathercpp.h"

fgElement* fgResource_Create(fgAsset res, const CRect* uv, unsigned int color, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement* r = fgroot_instance->backend.fgCreate("Resource", parent, next, name, flags, transform, units);
  if(color) _sendmsg<FG_SETCOLOR, size_t>(r, color);
  if(uv) _sendmsg<FG_SETUV, void*>(r, (void*)uv);
  if(res) _sendmsg<FG_SETASSET, void*>(r, res);
  return r;
}

void fgResource_Init(fgResource* self, fgElement* BSS_RESTRICT parent, fgElement* BSS_RESTRICT next, const char* name, fgFlag flags, const fgTransform* transform, unsigned short units)
{
  fgElement_InternalSetup(*self, parent, next, name, flags, transform, units, (fgDestroy)&fgResource_Destroy, (fgMessage)&fgResource_Message);
}

void fgResource_Destroy(fgResource* self)
{
  if(self->asset) fgroot_instance->backend.fgDestroyAsset(self->asset);
  self->asset = 0;
  self->element.message = (fgMessage)&fgElement_Message;
  fgElement_Destroy(&self->element);
}

void fgResource_Recalc(fgResource* self)
{
  if(self->asset && (self->element.flags&FGELEMENT_EXPAND))
  {
    AbsVec dim;
    fgroot_instance->backend.fgAssetSize(self->asset, &self->uv, &dim, self->element.flags);
    CRect adjust = self->element.transform.area;
    if(self->element.flags&FGELEMENT_EXPANDX)
      adjust.right.abs = adjust.left.abs + dim.x;
    if(self->element.flags&FGELEMENT_EXPANDY)
      adjust.bottom.abs = adjust.top.abs + dim.y;
    _sendmsg<FG_SETAREA, void*>(*self, &adjust);
  }
}

size_t fgResource_Message(fgResource* self, const FG_Msg* msg)
{
  assert(self != 0 && msg != 0);
  switch(msg->type)
  {
  case FG_CONSTRUCT:
    fgElement_Message(&self->element, msg);
    memset(&self->uv, 0, sizeof(CRect));
    self->uv.right.rel = 1.0f;
    self->uv.bottom.rel = 1.0f;
    self->color.color = 0;
    self->edge.color = 0;
    self->asset = 0;
    self->outline = 0;
    return FG_ACCEPT;
  case FG_SETUV:
    if(msg->p)
      self->uv = *((CRect*)msg->p);
    fgResource_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    return FG_ACCEPT;
  case FG_SETASSET:
    if(self->asset) fgroot_instance->backend.fgDestroyAsset(self->asset);
    self->asset = 0;
    if(msg->p) self->asset = fgroot_instance->backend.fgCloneAsset(msg->p, &self->element);
    fgResource_Recalc(self);
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_SETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_EDGE: self->edge.color = (uint32_t)msg->i; break;
    case FGSETCOLOR_MAIN: self->color.color = (uint32_t)msg->i; break;
    }
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_SETOUTLINE:
    self->outline = fgResolveUnit(&self->element, msg->f, msg->subtype, false);
    fgroot_instance->backend.fgDirtyElement(*self);
    break;
  case FG_GETUV:
    return (size_t)(&self->uv);
  case FG_GETASSET:
    return reinterpret_cast<size_t>((void*)self->asset);
  case FG_GETCOLOR:
    switch(msg->subtype)
    {
    case FGSETCOLOR_EDGE: return self->edge.color;
    case FGSETCOLOR_MAIN: return self->color.color;
    }
    return 0;
  case FG_GETOUTLINE:
    return *reinterpret_cast<size_t*>(&self->outline);
  case FG_MOVE:
    if(!(msg->u2 & FGMOVE_PROPAGATE) && (msg->u2 & FGMOVE_RESIZE))
      fgResource_Recalc(self);
    break;
  case FG_DRAW:
  {
    if(msg->subtype & 1) break;
    AbsRect area = *(AbsRect*)msg->p;
    fgDrawAuxData* data = (fgDrawAuxData*)msg->p2;
    fgScaleRectDPI(&area, data->dpi.x, data->dpi.y);
    fgSnapAbsRect(area, self->element.flags);
    AbsVec center = ResolveVec(&self->element.transform.center, &area);

    if(self->element.flags&FGRESOURCE_UVTILE)
    {
      self->uv.right.abs = self->uv.left.abs + area.right - area.left;
      self->uv.bottom.abs = self->uv.top.abs + area.bottom - area.top;
      fgroot_instance->backend.fgDrawAsset(self->asset, &self->uv, self->color.color, self->edge.color, self->outline, &area, self->element.transform.rotation, &center, self->element.flags, data);
    }
    else
      fgroot_instance->backend.fgDrawAsset(self->asset, &self->uv, self->color.color, self->edge.color, self->outline, &area, self->element.transform.rotation, &center, self->element.flags, data);
  }
  break;
  case FG_GETCLASSNAME:
    return (size_t)"Resource";
  }
  return fgElement_Message(&self->element, msg);
}

fgAsset fgCreateAssetFile(fgFlag flags, const char* file)
{
  FILE* f;
  FOPEN(f, file, "rb");
  if(!f) return 0;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  DYNARRAY(char, buf, len);
  fread(buf, 1, len, f);
  fclose(f);
  void* r = fgroot_instance->backend.fgCreateAsset(flags, buf, len);
  return r;
}