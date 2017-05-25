// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgBox.h"
#include "bss-util/bss_util.h"
#include "feathercpp.h"
#include <math.h>

size_t fgDefaultLayout(fgElement* self, const FG_Msg* msg, AbsVec* dim)
{
  assert(!msg->e || !(msg->e->flags & FGELEMENT_BACKGROUND));

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTREORDER:
  case FGELEMENT_LAYOUTRESIZE:
    break;
  case FGELEMENT_LAYOUTMOVE:
  {
    FG_Msg m = *msg;
    m.subtype = FGELEMENT_LAYOUTREMOVE;
    fgDefaultLayout(self, &m, dim);
    m.subtype = FGELEMENT_LAYOUTADD;
    fgDefaultLayout(self, &m, dim);
    return 0;
  }
  case FGELEMENT_LAYOUTADD:
    dim->x = fgLayout_ExpandX(dim->x, msg->e);
    dim->y = fgLayout_ExpandY(dim->y, msg->e);
    return 0;
  case FGELEMENT_LAYOUTREMOVE:
    assert(msg->e != 0);
    //if((child->transform.area.right.rel != child->transform.area.left.rel || // We recieve a LAYOUTMOVE message AFTER the child has already been moved. Because we have no idea where it used to be we can't use this optimization.
    //  child->transform.area.right.abs < dim->x) ||
    //  (child->transform.area.bottom.rel != child->transform.area.top.rel ||
    //    child->transform.area.bottom.abs < dim->y))
    //  break; // If there's no way the removed child could affect the layout dimensions, break.
  case FGELEMENT_LAYOUTRESET: // Otherwise, allow LAYOUTREMOVE to fall into LAYOUTRESET
  {
    fgElement* hold = self->root;
    dim->x = 0; // Reset area to nothing and then re-add everything.
    dim->y = 0;

    while(hold) // O(n) by necessity
    {
      if(hold != msg->e && !(hold->flags & FGELEMENT_BACKGROUND))
      {
        dim->x = fgLayout_ExpandX(dim->x, hold);
        dim->y = fgLayout_ExpandY(dim->y, hold);
      }
      hold = hold->next;
    }
    break;
  }
  }

  return 0;
}

BSS_FORCEINLINE FABS fgLayout_GetChildRight(fgElement* child)
{
  return child->transform.area.left.abs + fgLayout_GetElementWidth(child);
}

BSS_FORCEINLINE FABS fgLayout_GetChildBottom(fgElement* child)
{
  return child->transform.area.top.abs + fgLayout_GetElementHeight(child);
}

FABS fgTileGetExtra(fgElement* cur, char axis, FABS dim, FABS max)
{
  FABS x = (axis ? (cur->transform.area.bottom.rel - cur->transform.area.top.rel) : (cur->transform.area.right.rel - cur->transform.area.left.rel)) * max;
  if(dim > x) x = dim;
  if(axis && cur->maxdim.y >= 0.0f && x > cur->maxdim.y) x = cur->maxdim.y;
  if(!axis && cur->maxdim.x >= 0.0f && x > cur->maxdim.x) x = cur->maxdim.x;
  return x - dim;
}

void fgTileLayoutFill(fgElement* last, fgElement* skip, fgElement* row, FABS space, char axis, FABS max)
{
  assert(row != 0);
  if(space < 0) space = 0;
  FABS side = axis ? row->transform.area.top.abs : row->transform.area.left.abs;
  while(row != last)
  {
    if(row == skip || (row->flags&FGELEMENT_BACKGROUND)) // we check for cur background flags so we can pass in the root
    {
      row = row->next;
      continue;
    }

    // All relative tiles are restricted to expanding [space] units beyond their minimum size.
    FABS d = axis ? fgLayout_GetElementHeight(row) : fgLayout_GetElementWidth(row);
    FABS e = fgTileGetExtra(row, axis, d, max);
    if(axis && row->maxdim.y >= 0.0f && d > row->maxdim.y) d = row->maxdim.y;
    if(!axis && row->maxdim.x >= 0.0f && d > row->maxdim.x) d = row->maxdim.x;
    FABS y = bssmin(e, space);
    space -= y;
    if(axis)
    {
      row->transform.area.top.abs = side;
      row->transform.area.bottom.abs = (row->transform.area.bottom.rel != row->transform.area.top.rel) ? (side + d + y - max) : (side + d + y);
    }
    else
    {
      row->transform.area.left.abs = side;
      row->transform.area.right.abs = (row->transform.area.right.rel != row->transform.area.left.rel) ? (side + d + y - max) : (side + d + y);
    }
    side += d + y;
    row = row->next;
  }
}

AbsVec fgTileLayoutReorder(fgElement* cur, fgElement* skip, char axis, float max, AbsVec expand, AbsVec spacing, fgFlag flags) // axis: 0 means x-axis first, 1 means y-axis first
{
  if(axis)
  {
    std::swap(expand.x, expand.y);
    std::swap(spacing.x, spacing.y);
    flags = ((flags&FGBOX_IGNOREMARGINEDGEX) >> 1) | ((flags&FGBOX_IGNOREMARGINEDGEY) << 1);
  }
  FABS pitch = 0.0f;
  AbsVec pos = { 0,0 };
  while(cur && cur->flags&FGELEMENT_BACKGROUND) cur = cur->next;
  fgElement* row = cur;
  fgElement* prev;
  
  if(row != 0 && std::isfinite(max)) // If max is finite then we could potentially perform a tile fill and will require pitch, so find the start of our row and initialize pitch.
  {
    FABS start = (axis ? row->transform.area.left.abs : row->transform.area.top.abs);
    FABS dim;
    prev = row;
    while(row != 0 && (axis ? row->transform.area.left.abs : row->transform.area.top.abs) == start)
    {
      dim = (axis ? fgLayout_GetChildRight(row) : fgLayout_GetChildBottom(row)) - start;
      if(flags&FGBOX_IGNOREMARGINEDGEY)
        dim -= (axis ? cur->margin.left : cur->margin.top);
      if(dim > pitch)
        pitch = dim;
      prev = row;
      row = fgLayout_GetPrev(row); // run backwards until we go up a level
    }
    row = prev;
    pitch += spacing.y;
  }
  if((prev = (!cur ? 0 : fgLayout_GetPrev(cur))))
    pos = axis ? AbsVec{ fgLayout_GetChildBottom(prev), prev->transform.area.left.abs } : AbsVec{ fgLayout_GetChildRight(prev), prev->transform.area.top.abs };
  if(cur != 0 && (flags&FGBOX_IGNOREMARGINEDGEX))
    pos.x -= axis ? cur->margin.top : cur->margin.left;
  fgElement* firstrow = row;

  while(cur)
  {
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND))
    {
      cur = cur->next;
      continue;
    }
    AbsVec dim = axis ? AbsVec{ fgLayout_GetElementHeight(cur), fgLayout_GetElementWidth(cur) } : AbsVec{ fgLayout_GetElementWidth(cur), fgLayout_GetElementHeight(cur) };

    if(row != cur && pos.x + dim.x > max)
    {
      fgTileLayoutFill(cur, skip, row, max - pos.x, axis, max);
      pos.y += pitch;
      pos.x = (flags&FGBOX_IGNOREMARGINEDGEX) ? -(axis ? cur->margin.top : cur->margin.left) : 0;
      pitch = 0;
      row = cur;
    }
    FABS posy = (row == firstrow && (flags&FGBOX_IGNOREMARGINEDGEY)) ? pos.y - (axis ? cur->margin.left : cur->margin.top) : pos.y;

    if(!axis)
      MoveCRect(pos.x, posy, &cur->transform.area);
    else
      MoveCRect(posy, pos.x, &cur->transform.area);

    pos.x += dim.x + spacing.x;
    dim.y += posy - pos.y;
    if(dim.y > pitch) pitch = dim.y;
    dim.y += pos.y;
    if(pos.x > expand.x) expand.x = pos.x;
    if(dim.y > expand.y) expand.y = dim.y;
    cur = cur->next;
  }

  if(row != 0 && std::isfinite(max))
    fgTileLayoutFill(0, skip, row, max - pos.x, axis, max);
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  return expand;
}

void fgTileRecalcMin(fgElement* self, const FG_Msg* msg, char axis)
{
  fgElement* child = msg->e;
  fgElement* cur = self->root;
  FABS w;
  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTRESET:
  case FGELEMENT_LAYOUTMOVE:
    child = 0;
  case FGELEMENT_LAYOUTREMOVE:
    if(!axis)
      self->mindim.x = 0;
    else
      self->mindim.y = 0;

    while(cur)
    {
      if(cur != child && !(cur->flags&FGELEMENT_BACKGROUND))
      {
        if(!axis && (w = fgLayout_GetElementWidth(cur) + self->padding.left + self->padding.right + self->margin.left + self->margin.right) > self->mindim.x) self->mindim.x = w;
        if(axis && (w = fgLayout_GetElementHeight(cur) + self->padding.top + self->padding.bottom + self->margin.top + self->margin.bottom) > self->mindim.y) self->mindim.y = w;
      }
      cur = cur->next;
    }
    break;
  case FGELEMENT_LAYOUTADD:
    if(!axis && (w = fgLayout_GetElementWidth(child) + self->padding.left + self->padding.right + self->margin.left + self->margin.right) > self->mindim.x) self->mindim.x = w;
    if(axis && (w = fgLayout_GetElementHeight(child) + self->padding.top + self->padding.bottom + self->margin.top + self->margin.bottom) > self->mindim.y) self->mindim.y = w;
    break;
  }
}

size_t fgTileLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* dim, AbsVec spacing)
{
  AbsVec curdim = *dim;
  fgElement* child = msg->e;
  fgElement* skip = child;
  char axis = ((flags & FGBOX_TILE) && (flags & FGBOX_GROWY)) || ((flags & FGBOX_TILEY) && !(flags & FGBOX_TILEX)); // 0 expands along x-axis, 1 expands along y-axis
  FABS max = INFINITY; // Maximum axis length. If axis is 0, represents maximum length along x axis, otherwise represents maximum length along y-axis.
  if((flags & FGBOX_TILE) == FGBOX_TILE)
  {
    if((!axis && self->transform.area.left.rel != self->transform.area.right.rel) || (axis && self->transform.area.top.rel != self->transform.area.bottom.rel))
      fgTileRecalcMin(self, msg, axis);

    AbsRect out;
    ResolveInnerRect(self, &out);
    AbsVec selfdim = { out.right - out.left, out.bottom - out.top };

    max = !axis ? ((self->flags & FGELEMENT_EXPANDX) ? self->maxdim.x : selfdim.x) : ((self->flags & FGELEMENT_EXPANDY) ? self->maxdim.y : selfdim.y);
    if(max < 0) max = INFINITY;
  }

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTRESET:
    curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec { 0,0 }, spacing, flags);
    break;
  case FGELEMENT_LAYOUTRESIZE: // we only need to resize if the tiling is wrapping (flags is set to FGBOX_TILE) and the limiting axis is the same one that was resized.
    if((((msg->u2 & FGMOVE_RESIZEX) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->u2 & FGMOVE_RESIZEY) && !(self->flags&FGELEMENT_EXPANDY))) && (flags & FGBOX_TILE))
      curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec{ 0,0 }, spacing, flags);
    else
      return 0;
    break;
  case FGELEMENT_LAYOUTREORDER:
    if((flags&FGBOX_TILE) == FGBOX_TILE) // If the tiling is wrapping we have to recalculate everything
      curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec{ 0,0 }, spacing, flags);
    else
    {
      fgElement* old = msg->e2; // Both of these could temporarily be BACKGROUND elements, so we can't assert that.
      fgElement* cur = old;
      while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
      curdim = fgTileLayoutReorder(!cur ? child : old, 0, axis, max, curdim, spacing, flags);
    }
    break;
  case FGELEMENT_LAYOUTADD:
    curdim = fgTileLayoutReorder(child, 0, axis, max, curdim, spacing, flags);
    break;
  case FGELEMENT_LAYOUTMOVE:
    if(!(msg->u2&FGMOVE_RESIZE)) // Only reorder if the child was resized. Note that we need to resize even if it's the opposite axis to account for expansion
      return 0;
    skip = 0;
  case FGELEMENT_LAYOUTREMOVE: // If we are expanding along an axis that isn't tiled, or we are wrapping, we have to recalculate the whole thing.
    if(((self->flags&FGBOX_TILE) == FGBOX_TILE) || ((self->flags&FGELEMENT_EXPANDX) && !(flags&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(flags&FGBOX_TILEY)))
      curdim = fgTileLayoutReorder(self->root, skip, axis, max, AbsVec { 0,0 }, spacing, flags);
    else
    {
      fgElement* prev = fgLayout_GetPrev(child);
      if(!axis)
        curdim.x = !prev ? 0 : fgLayout_GetChildRight(prev);
      else
        curdim.y = !prev ? 0 : fgLayout_GetChildBottom(prev);
      curdim = fgTileLayoutReorder(child, skip, axis, max, curdim, spacing, flags);
    }
    break;
  }

  *dim = curdim;
  return 0;
}

AbsVec fgDistributeLayoutReorder(fgElement* prev, fgElement* cur, fgElement* skip, char axis, AbsVec expand, AbsRect* cache, AbsRect* padding) // axis: 0 means x-axis first, 1 means y-axis first
{
  float cursor = 0.0f;
  if(prev)
  {
    assert(!(prev->flags&FGELEMENT_BACKGROUND));
    cursor = axis ? fgLayout_GetChildBottom(prev) : fgLayout_GetChildRight(prev);
  }
  while(cur)
  {
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND)) // we check for cur background flags so we can pass in the root
    {
      cur = cur->next;
      continue;
    }
    CRect area = cur->transform.area;
    AbsRect abs;
    ResolveRectCache(cur, &abs, cache, padding);
    if(axis)
    {
      area.top.abs = cursor;
      area.bottom.abs = cursor;
      area.bottom.rel -= area.top.rel;
      area.top.rel = 0;
      cursor += abs.bottom - abs.top;
      if(cursor > expand.y) expand.y = cursor;
      if(abs.right > expand.x) expand.x = abs.right;
    }
    else
    {
      area.left.abs = cursor;
      area.right.abs = cursor;
      area.right.rel -= area.left.rel;
      area.left.rel = 0;
      cursor += abs.right - abs.left;
      if(cursor > expand.x) expand.x = cursor;
      if(abs.bottom > expand.y) expand.y = abs.bottom;
    }
    cur->SetArea(area);
    cur = cur->next;
  }

  return expand;
}
size_t fgDistributeLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* dim)
{
  AbsRect cache;
  ResolveRect(self, &cache);
  fgElement* child = msg->e;
  char axis = (flags&FGBOX_GROWY); // 0 expands along x-axis, 1 expands along y-axis
  switch(msg->type)
  {
  case FGELEMENT_LAYOUTRESET:
  case FGELEMENT_LAYOUTMOVE: // we don't handle moving or resizing because we use relative coordinates, so the resize is done for us.
  case FGELEMENT_LAYOUTRESIZE:
  case FGELEMENT_LAYOUTREORDER:
    *dim = fgDistributeLayoutReorder(0, self->root, 0, axis, AbsVec{ 0,0 }, &cache, &self->padding);
    break;
  case FGELEMENT_LAYOUTADD:
    *dim = fgDistributeLayoutReorder(fgLayout_GetPrev(child), child, 0, axis, *dim, &cache, &self->padding);
    break;
  case FGELEMENT_LAYOUTREMOVE:
    *dim = fgDistributeLayoutReorder(0, self->root, child, axis, AbsVec{ 0,0 }, &cache, &self->padding);
    break;
  }

  return 0;
}