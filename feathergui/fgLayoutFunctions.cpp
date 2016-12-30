// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayout.h"
#include "fgBox.h"
#include "bss-util/bss_util.h"
#include "feathercpp.h"

#include <math.h>

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetElementMinWidth(fgElement* child)
{
  if(!(child->flags&FGELEMENT_EXPANDX))
    return child->mindim.x;
  FABS l = child->layoutdim.x + child->padding.left + child->padding.right + child->margin.left + child->margin.right;
  return (child->mindim.x >= 0.0f && child->mindim.x > l) ? child->mindim.x : l;
}

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetElementMinHeight(fgElement* child)
{
  if(!(child->flags&FGELEMENT_EXPANDY))
    return child->mindim.y;
  FABS l = child->layoutdim.y + child->padding.top + child->padding.bottom + child->margin.top + child->margin.bottom;
  return (child->mindim.y >= 0.0f && child->mindim.y > l) ? child->mindim.y : l;
}

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetElementWidth(fgElement* child)
{
  FABS w = (child->transform.area.left.rel == child->transform.area.right.rel) ? child->transform.area.right.abs - child->transform.area.left.abs : 0.0f;
  FABS m = fgLayout_GetElementMinWidth(child);
  return (m >= 0.0f && m > w) ? m : w;
}

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetElementHeight(fgElement* child)
{
  FABS h = (child->transform.area.top.rel == child->transform.area.bottom.rel) ? child->transform.area.bottom.abs - child->transform.area.top.abs : 0.0f;
  FABS m = fgLayout_GetElementMinHeight(child);
  return (m >= 0.0f && m > h) ? m : h;
}

inline FABS FG_FASTCALL fgLayout_ExpandX(FABS dimx, fgElement* child)
{
  FABS r = child->transform.area.left.abs + fgLayout_GetElementWidth(child);
  return bssmax(dimx, r);
}

inline FABS FG_FASTCALL fgLayout_ExpandY(FABS dimy, fgElement* child)
{
  FABS r = child->transform.area.top.abs + fgLayout_GetElementHeight(child);
  return bssmax(dimy, r);
}

size_t FG_FASTCALL fgDefaultLayout(fgElement* self, const FG_Msg* msg, AbsVec* dim)
{
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
    fgDefaultLayout(self, &m, dim);
    m.subtype = FGELEMENT_LAYOUTADD;
    fgDefaultLayout(self, &m, dim);
    return 0;
  }
  case FGELEMENT_LAYOUTADD:
    dim->x = fgLayout_ExpandX(dim->x, child);
    dim->y = fgLayout_ExpandY(dim->y, child);
    return 0;
  case FGELEMENT_LAYOUTREMOVE:
    assert(child != 0);
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
      if(hold != child)
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

BSS_FORCEINLINE fgElement* fgLayout_GetNext(fgElement* cur)
{
  while((cur = cur->next) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

BSS_FORCEINLINE fgElement* fgLayout_GetPrev(fgElement* cur)
{
  while((cur = cur->prev) != 0 && (cur->flags&FGELEMENT_BACKGROUND) != 0);
  return cur;
}

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetChildRight(fgElement* child)
{
  return child->transform.area.left.abs + fgLayout_GetElementWidth(child);
}

BSS_FORCEINLINE FABS FG_FASTCALL fgLayout_GetChildBottom(fgElement* child)
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

void FG_FASTCALL fgTileLayoutFill(fgElement* cur, fgElement* skip, fgElement* row, FABS space, char axis, FABS max, FABS extra)
{
  FABS side = max;
  if(extra < space) // if this is true we don't actually have enough to fill the space in, so we must make adjustments
  {
    side -= space - extra;
    space = extra;
  }

  while(cur != row)
  {
    assert(cur != 0);
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND)) // we check for cur background flags so we can pass in the root
    {
      cur = cur->prev;
      continue;
    }

    // All relative tiles are restricted to expanding [space] units beyond their minimum size.
    FABS d = axis ? fgLayout_GetElementHeight(cur) : fgLayout_GetElementWidth(cur);
    FABS e = fgTileGetExtra(cur, axis, d, max);
    if(axis && cur->maxdim.y >= 0.0f && d > cur->maxdim.y) d = cur->maxdim.y;
    if(!axis && cur->maxdim.x >= 0.0f && d > cur->maxdim.x) d = cur->maxdim.x;
    FABS y = bssmin(e, space);
    space -= y;
    if(axis)
    {
      cur->transform.area.top.abs = side - d - y;
      cur->transform.area.bottom.abs = (cur->transform.area.bottom.rel != cur->transform.area.top.rel) ? side - max : side;
    }
    else
    {
      cur->transform.area.left.abs = side - d - y;
      cur->transform.area.right.abs = (cur->transform.area.right.rel != cur->transform.area.left.rel) ? side - max : side;
    }
    side = side - d - y;
    cur = cur->prev;
  }
}

AbsVec fgTileLayoutReorder(fgElement* cur, fgElement* skip, char axis, float max, AbsVec expand) // axis: 0 means x-axis first, 1 means y-axis first
{
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  FABS pitch = 0.0f;
  FABS extra = 0.0f;
  AbsVec pos = { 0,0 };
  fgElement* prev = !cur ? 0 : fgLayout_GetPrev(cur);
  fgElement* row = cur;
  
  if(std::isfinite(max)) // If max is finite then we could potentially perform a tile fill and will require pitch, so find the start of our row and initialize pitch.
  {
    FABS start = (axis ? row->transform.area.left.abs : row->transform.area.top.abs);
    FABS dim;
    while(row != 0 && (axis ? row->transform.area.left.abs : row->transform.area.top.abs) == start)
    {
      dim = (axis ? fgLayout_GetChildRight(row) : fgLayout_GetChildBottom(row)) - start;
      if(dim > pitch)
        pitch = dim;
      if(row != cur) // don't double-count the extra space
        extra += fgTileGetExtra(row, axis, dim, max);
      row = fgLayout_GetPrev(row); // run backwards until we go up a level
    }
  }
  if(prev)
    pos = axis ? AbsVec{ fgLayout_GetChildBottom(prev), prev->transform.area.left.abs } : AbsVec{ fgLayout_GetChildRight(prev), prev->transform.area.top.abs };

  while(cur)
  {
    if(cur == skip || (cur->flags&FGELEMENT_BACKGROUND))
    {
      cur = cur->next;
      continue;
    }
    AbsVec dim = axis ? AbsVec{ fgLayout_GetElementHeight(cur), fgLayout_GetElementWidth(cur) } : AbsVec{ fgLayout_GetElementWidth(cur), fgLayout_GetElementHeight(cur) };

    if(pos.x + dim.x > max)
    {
      fgTileLayoutFill(prev, skip, row, max - pos.x, axis, max, extra);
      pos.y += pitch;
      pos.x = 0;
      pitch = 0;
      row = prev;
    }

    if(!axis)
      MoveCRect(pos.x, pos.y, &cur->transform.area);
    else
      MoveCRect(pos.y, pos.x, &cur->transform.area);

    if(std::isfinite(max)) extra += fgTileGetExtra(cur, axis, dim.x, max);
    pos.x += dim.x;
    if(dim.y > pitch) pitch = dim.y;
    dim.y += pos.y;
    if(pos.x > expand.x) expand.x = pos.x;
    if(dim.y > expand.y) expand.y = dim.y;
    prev = cur;
    cur = cur->next;
  }

  if(std::isfinite(max))
    fgTileLayoutFill(prev, skip, row, max - pos.x, axis, max, extra);
  if(axis) { FABS t = expand.x; expand.x = expand.y; expand.y = t; }
  return expand;
}

size_t FG_FASTCALL fgTileLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* dim)
{
  AbsVec curdim = *dim;
  fgElement* child = (fgElement*)msg->other;
  char axis = ((flags & FGBOX_TILE) && (flags & FGBOX_DISTRIBUTEY)) || ((flags & FGBOX_TILEY) && !(flags & FGBOX_TILEX)); // 0 expands along x-axis, 1 expands along y-axis
  FABS max = INFINITY; // Maximum axis length. If axis is 0, represents maximum length along x axis, otherwise represents maximum length along y-axis.
  AbsRect out;
  ResolveInnerRect(self, &out);
  AbsVec selfdim = { out.right - out.left, out.bottom - out.top };
  if((flags & FGBOX_TILE) == FGBOX_TILE)
  {
    max = !axis ? ((self->flags & FGELEMENT_EXPANDX) ? self->maxdim.x : selfdim.x) : ((self->flags & FGELEMENT_EXPANDY) ? self->maxdim.y : selfdim.y);
    if(max < 0) max = INFINITY;
  }

  switch(msg->subtype)
  {
  case FGELEMENT_LAYOUTRESET:
    curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec { 0,0 });
    break;
  case FGELEMENT_LAYOUTRESIZE: // we only need to resize if the tiling is wrapping (flags is set to FGBOX_TILE) and the limiting axis is the same one that was resized.
    if((((msg->otheraux & FGMOVE_RESIZEX) && !(self->flags&FGELEMENT_EXPANDX)) || ((msg->otheraux & FGMOVE_RESIZEY) && !(self->flags&FGELEMENT_EXPANDY))) && (flags & FGBOX_TILE))
      curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec{ 0,0 });
    else
      return 0;
    break;
  case FGELEMENT_LAYOUTREORDER:
    if((flags&FGBOX_TILE) == FGBOX_TILE) // If the tiling is wrapping we have to recalculate everything
      curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec{ 0,0 });
    else
    {
      fgElement* old = (fgElement*)msg->other2;
      fgElement* cur = old;
      assert(!(old->flags&FGELEMENT_BACKGROUND));
      assert(!(cur->flags&FGELEMENT_BACKGROUND));
      while(cur != 0 && cur != child) cur = cur->next; // Run down from old until we either hit child, in which case old is the lowest, or we hit null
      curdim = fgTileLayoutReorder(!cur ? child : old, 0, axis, max, curdim);
    }
    break;
  case FGELEMENT_LAYOUTMOVE:
    if(!(msg->otheraux&FGMOVE_RESIZE)) // Only reorder if the child was resized. Note that we need to resize even if it's the opposite axis to account for expansion
      return 0;
    if(((self->flags&FGBOX_TILE) == FGBOX_TILE) || ((self->flags&FGELEMENT_EXPANDX) && !(flags&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(flags&FGBOX_TILEY)))
      curdim = fgTileLayoutReorder(self->root, 0, axis, max, AbsVec{ 0,0 });
    else
    {
      fgElement* prev = fgLayout_GetPrev(child);
      if(!axis)
        curdim.x = !prev ? 0 : fgLayout_GetChildRight(prev);
      else
        curdim.y = !prev ? 0 : fgLayout_GetChildBottom(prev);
      curdim = fgTileLayoutReorder(child, 0, axis, max, curdim);
    }
    break;
  case FGELEMENT_LAYOUTADD:
    curdim = fgTileLayoutReorder(child, 0, axis, max, curdim);
    break;
  case FGELEMENT_LAYOUTREMOVE: // If we are expanding along an axis that isn't tiled, or we are wrapping, we have to recalculate the whole thing.
    if(((self->flags&FGBOX_TILE) == FGBOX_TILE) || ((self->flags&FGELEMENT_EXPANDX) && !(flags&FGBOX_TILEX)) || ((self->flags&FGELEMENT_EXPANDY) && !(flags&FGBOX_TILEY)))
      curdim = fgTileLayoutReorder(self->root, child, axis, max, AbsVec { 0,0 });
    else
    {
      fgElement* prev = fgLayout_GetPrev(child);
      if(!axis)
        curdim.x = !prev ? 0 : fgLayout_GetChildRight(prev);
      else
        curdim.y = !prev ? 0 : fgLayout_GetChildBottom(prev);
      curdim = fgTileLayoutReorder(child, child, axis, max, curdim);
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
size_t FG_FASTCALL fgDistributeLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* dim)
{
  AbsRect cache;
  ResolveRect(self, &cache);
  fgElement* child = (fgElement*)msg->other;
  char axis = (flags&FGBOX_DISTRIBUTEY); // 0 expands along x-axis, 1 expands along y-axis
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