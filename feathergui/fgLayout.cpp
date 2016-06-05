// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgButton.h"
#include "fgWindow.h"
#include "fgText.h"
#include "fgResource.h"
#include "feathercpp.h"
#include "fgBox.h"

void FG_FASTCALL fgLayout_Init(fgLayout* self)
{
  memset(self, 0, sizeof(fgLayout));
}
void FG_FASTCALL fgLayout_Destroy(fgLayout* self)
{
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  reinterpret_cast<fgClassLayoutArray&>(self->layout).~cArraySort();
}
FG_UINT FG_FASTCALL fgLayout_AddResource(fgLayout* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgLayout_RemoveResource(fgLayout* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgLayout_GetResource(fgLayout* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
FG_UINT FG_FASTCALL fgLayout_AddFont(fgLayout* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgLayout_RemoveFont(fgLayout* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgLayout_GetFont(fgLayout* self, FG_UINT font)
{
  return ((fgFontArray&)self->fonts)[font].ref;
}
FG_UINT FG_FASTCALL fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  return ((fgClassLayoutArray&)self->layout).Insert(fgClassLayoutConstruct(type, name, flags, transform, order));
}
char FG_FASTCALL fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout)
{
  return DynArrayRemove((fgClassLayoutArray&)self->layout, layout);
}
fgClassLayout* FG_FASTCALL fgLayout_GetLayout(fgLayout* self, FG_UINT layout)
{
  return self->layout.p + layout;
}

void FG_FASTCALL fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  fgStyleLayout_Init(&self->style, type, name, flags, transform, order);
  memset(&self->children, 0, sizeof(fgVector));
}
void FG_FASTCALL fgClassLayout_Destroy(fgClassLayout* self)
{
  fgStyleLayout_Destroy(&self->style);
  reinterpret_cast<fgClassLayoutArray&>(self->children).~cArraySort();
}

FG_UINT FG_FASTCALL fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  return ((fgClassLayoutArray&)self->children).Insert(fgClassLayoutConstruct(type, name, flags, transform, order));
}
char FG_FASTCALL fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgClassLayout* FG_FASTCALL fgClassLayout_GetChild(fgClassLayout* self, FG_UINT child)
{
  return self->children.p + child;
}

inline char FG_FASTCALL fgLayout_ExpandX(CRect* selfarea, fgElement* child, FABS dimx)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->right.abs - selfarea->left.abs;
  if(area->right.abs > d)
  {
    selfarea->right.abs = selfarea->left.abs + area->right.abs;
    return FGMOVE_RESIZEX;
  }
  return 0;
}

inline char FG_FASTCALL fgLayout_ExpandY(CRect* selfarea, fgElement* child, FABS dimy)
{
  CRect* area = &child->transform.area;
  FABS d = selfarea->bottom.abs - selfarea->top.abs;
  if(area->bottom.abs > d)
  {
    selfarea->bottom.abs = selfarea->top.abs + area->bottom.abs;
    return FGMOVE_RESIZEY;
  }
  return 0;
}

size_t FG_FASTCALL fgLayout_Default(fgElement* self, const FG_Msg* msg, CRect* area, AbsRect* parent)
{
  if(!(self->flags & FGELEMENT_EXPAND))
    return 0;

  fgElement* child = (fgElement*)msg->other;
  assert(!child || !(child->flags & FGELEMENT_BACKGROUND));

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTREORDER:
  case FGELEMENT_LAYOUTRESIZE:
    break;
  case FGELEMENT_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGELEMENT_LAYOUTREMOVE;
    size_t dim = fgLayout_Default(self, &m, area, parent);
    m.subtype = FGELEMENT_LAYOUTADD;
    dim |= fgLayout_Default(self, &m, area, parent);
    return dim;
  }
  case FGELEMENT_LAYOUTADD:
  {
    char dim = 0;
    if(self->flags & FGELEMENT_EXPANDX)
      dim |= fgLayout_ExpandX(area, child, parent->right - parent->left);
    if(self->flags & FGELEMENT_EXPANDY)
      dim |= fgLayout_ExpandY(area, child, parent->bottom - parent->top);
    return dim;
  }
  case FGELEMENT_LAYOUTREMOVE:
  {
    CRect* childarea = &child->transform.area;
    FABS dx = self->transform.area.right.abs - self->transform.area.left.abs;
    FABS dy = self->transform.area.bottom.abs - self->transform.area.top.abs;
    if(((self->flags & FGELEMENT_EXPANDX) &&
      childarea->right.rel == childarea->left.rel &&
      childarea->right.abs >= dx) ||
      ((self->flags & FGELEMENT_EXPANDY) &&
        childarea->bottom.rel == childarea->top.rel &&
        childarea->bottom.abs >= dy))
    {
      FG_Msg m = *msg;
      m.subtype = FGELEMENT_LAYOUTRESET;
      m.other = child;
      return fgLayout_Default(self, &m, area, parent);
    }
  }
  case FGELEMENT_LAYOUTRESET:
  {
    fgElement* hold = self->root;
    if(self->flags & FGELEMENT_EXPANDX)
      area->right.abs = area->left.abs; // Reset area to smallest possible, then re-add everything other than the removed child
    if(self->flags & FGELEMENT_EXPANDY)
      area->bottom.abs = area->top.abs;
    char dim = 0;

    while(hold) // O(n) by necessity
    {
      if(hold != child)
      {
        if(self->flags & FGELEMENT_EXPANDX)
          dim |= fgLayout_ExpandX(area, hold, parent->right - parent->left);
        if(self->flags & FGELEMENT_EXPANDY)
          dim |= fgLayout_ExpandY(area, hold, parent->bottom - parent->top);
      }
      hold = hold->next;
    }

    return dim;
  }
  }

  return 0;
}

char fgLayout_TileSame(FABS posx, FABS posy, fgElement* cur)
{
  return (posx > 0 || posy > 0) && posx == cur->transform.area.left.abs && posy == cur->transform.area.left.abs;
}

fgElement* fgLayout_GetNext(fgElement* cur)
{
  while((cur = cur->next) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

fgElement* fgLayout_GetPrev(fgElement* cur)
{
  while((cur = cur->prev) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

AbsVec fgLayout_TileReorder(fgElement* prev, fgElement* cur, fgElement* skip, char axis, float max, float pitch, AbsVec expand) // axis: 0 means x-axis first, 1 means y-axis first
{
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  AbsVec pos = { 0,0 };
  if(prev)
  {
    assert(!(prev->flags&FGELEMENT_BACKGROUND));
    pos.x = axis ? prev->transform.area.bottom.abs : prev->transform.area.right.abs;
    pos.y = axis ? prev->transform.area.left.abs : prev->transform.area.top.abs;
  }
  char rowbump = 0; // Used to keep track if we've gone down at least one row - this is required for our same position optimization to be valid.

  while(cur) // O(n) complexity, but should break after re-ordering becomes unnecessary.
  {
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND)) // we check for cur background flags so we can pass in the root
    {
      cur = cur->next;
      continue;
    }
    if(fgLayout_TileSame(axis ? pos.y : pos.x, axis ? pos.x : pos.y, cur) && rowbump)
      break;
    AbsVec dim = { (cur->transform.area.right.abs - cur->transform.area.left.abs), (cur->transform.area.bottom.abs - cur->transform.area.top.abs) };
    if(axis) { FABS t = dim.x; dim.x = dim.y; dim.y = t; }

    if(pos.x + dim.x > max)
    {
      pos.y += pitch;
      pos.x = 0;
      pitch = 0;
      rowbump = 1;
    }

    if(!axis)
      MoveCRect(pos.x, pos.y, &cur->transform.area);
    else
      MoveCRect(pos.y, pos.x, &cur->transform.area);
    pos.x += dim.x;
    if(dim.y > pitch) pitch = dim.y;
    dim.y += pos.y;
    if(pos.x > expand.x) expand.x = pos.x;
    if(dim.y > expand.y) expand.y = dim.y;
    cur = cur->next;
  }
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  return expand;
}

FABS FG_FASTCALL fgLayout_TileGetPitch(fgElement* cur, char axis)
{
  if(!cur) return 0.0f;
  FABS start = (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs);
  FABS dim;
  FABS pitch = 0.0;
  while(cur != 0 && (axis ? cur->transform.area.left.abs : cur->transform.area.top.abs) == start)
  {
    dim = (axis ? cur->transform.area.right.abs : cur->transform.area.bottom.abs) - start;
    if(dim > pitch)
      pitch = dim;
    cur = fgLayout_GetPrev(cur); // run backwards until we go up a level
  }
  return pitch;
}

size_t FG_FASTCALL fgLayout_Tile(fgElement* self, const FG_Msg* msg, fgFlag axes, CRect* area)
{
  AbsVec realsize = AbsVec { area->right.abs - area->left.abs, area->bottom.abs - area->top.abs };
  fgElement* child = (fgElement*)msg->other;
  fgElement* prev;
  char axis = ((axes & FGBOX_TILE) && (axes & FGBOX_DISTRIBUTEY)) || ((axes & FGBOX_TILEY) && !(axes & FGBOX_TILEX));
  FABS max = INFINITY;
  if(axes & FGBOX_TILE)
  {
    AbsRect out;
    ResolveRect(self, &out);
    if((axes & FGBOX_DISTRIBUTEY) && !(self->flags & FGELEMENT_EXPANDY)) max = out.bottom - out.top;
    else if(!(self->flags & FGELEMENT_EXPANDX)) max = out.right - out.left;
  }

  AbsVec expand;
  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTRESET:
    expand = fgLayout_TileReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
    break;
  case FGELEMENT_LAYOUTRESIZE:
    if((((msg->otheraux & FGMOVE_RESIZEX) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->otheraux & FGMOVE_RESIZEY) && !(self->flags&FGELEMENT_EXPANDY))) && (axes & FGBOX_TILE))
      expand = fgLayout_TileReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
    else
      return 0;
    break;
  case FGELEMENT_LAYOUTREORDER:
  {
    fgElement* old = (fgElement*)msg->other2;
    fgElement* cur = old;
    assert(!(old->flags&FGELEMENT_BACKGROUND));
    assert(!(cur->flags&FGELEMENT_BACKGROUND));
    while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
    child = !cur ? child : old;

    prev = fgLayout_GetPrev(child); // walk backwards until we hit the previous non-background element
    expand = fgLayout_TileReorder(prev, child, 0, axis, max, (axes & FGBOX_TILE) ? fgLayout_TileGetPitch(prev, axis) : 0.0f, realsize);
  }
  break;
  case FGELEMENT_LAYOUTMOVE:
    if(!(msg->otheraux&FGMOVE_RESIZE)) // Only reorder if the child was resized. Note that we need to resize even if it's the opposite axis to account for expansion
      return 0;
    if(((self->flags&FGELEMENT_EXPANDX) && !(axes&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(axes&FGBOX_TILEY))) 
    { // If we are expanding along an axis that isn't tiled, we have to recalculate the whole thing.
      expand = fgLayout_TileReorder(0, self->root, 0, axis, max, 0.0f, AbsVec { 0,0 });
      break;
    }

    prev = fgLayout_GetPrev(child);
    if(prev)
      realsize = AbsVec { prev->transform.area.right.abs, prev->transform.area.bottom.abs };
    else
      realsize = AbsVec { 0,0 };

    prev = fgLayout_GetPrev(child);
    expand = fgLayout_TileReorder(prev, child, 0, axis, max, (axes & FGBOX_TILE) ? fgLayout_TileGetPitch(prev, axis) : 0.0f, realsize);
    break;
  case FGELEMENT_LAYOUTADD:
    prev = fgLayout_GetPrev(child);
    expand = fgLayout_TileReorder(prev, child, 0, axis, max, (axes & FGBOX_TILE) ? fgLayout_TileGetPitch(prev, axis) : 0.0f, realsize);
    break;
  case FGELEMENT_LAYOUTREMOVE:
    if(((self->flags&FGELEMENT_EXPANDX) && !(axes&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(axes&FGBOX_TILEY)))
    { // If we are expanding along an axis that isn't tiled, we have to recalculate the whole thing.
      expand = fgLayout_TileReorder(0, self->root, child, axis, max, 0.0f, AbsVec { 0,0 });
      break;
    }

    prev = fgLayout_GetPrev(child);
    if(prev)
      realsize = AbsVec { prev->transform.area.right.abs, prev->transform.area.bottom.abs };
    else
      realsize = AbsVec { 0,0 };

    expand = fgLayout_TileReorder(prev, fgLayout_GetNext(child), 0, axis, max, (axes & FGBOX_TILE) ? fgLayout_TileGetPitch(prev, axis) : 0.0f, realsize);
    break;
  }

  char dim = 0; // We always expand the area regardless of the EXPAND flags because the function that calls this should use the information appropriately.
  if(area->right.abs - area->left.abs != expand.x)
  {
    area->right.abs = area->left.abs + expand.x;
    dim |= FGMOVE_RESIZEX;
  }
  if(area->bottom.abs - area->top.abs != expand.y)
  {
    area->bottom.abs = area->top.abs + expand.y;
    dim |= FGMOVE_RESIZEY;
  }
  return dim;
}

size_t FG_FASTCALL fgLayout_Distribute(fgElement* self, const FG_Msg* msg, fgFlag axis)
{
  /*switch(msg->type)
  {
  case FGELEMENT_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGELEMENT_LAYOUTRESIZE:
  break;
  case FGELEMENT_LAYOUTREORDER:
  //child = child->order<msg->other->order?child:msg->other; // Get lowest child
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTADD:
  fgLayout_DistributeReorder(child->prev, child, axis, num);
  break;
  case FGELEMENT_LAYOUTREMOVE:
  fgLayout_DistributeReorder(child->prev, child->next, axis, num);
  break;
  }*/

  return 0;
}