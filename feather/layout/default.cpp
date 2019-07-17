// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/layout/default.h"
#include "feather/rtree.h"
#include "../util.h"
#include <vector>
#include <string>
#include <assert.h>

fgRect fgResolveArea(const URect& in, const UVec& center, const fgRect& resolvedParentArea)
{
  // The area relative coordinates are relative to the resolved area of our parent node.
  fgVec parentSize = { resolvedParentArea.right - resolvedParentArea.left, resolvedParentArea.bottom - resolvedParentArea.top };

  fgRect area = {
    (in.rel.left * parentSize.x),
    (in.rel.top * parentSize.y),
    (in.rel.right * parentSize.x),
    (in.rel.bottom * parentSize.y),
  };
  area += in.abs;

  // The center offset is relative to our own resolved area, which is what allows perfect centering.
  fgVec size = { area.right - area.left, area.bottom - area.top };
  fgVec offset = { (center.rel.x * size.x), (center.rel.y * size.y) };
  offset += center.abs;

  area.left -= offset.x;
  area.top -= offset.y;
  area.right -= offset.x;
  area.bottom -= offset.y;

  return area;
}

void fgLayoutDefault(fgDocumentNode* node, const fgRect* area, const fgOutlineNode* parent, float scale, fgVec dpi)
{
  static const int MAXNODES = 4;
  assert(node && node->outline);

  if(parent != 0) // If we've sent in the parent, this node is being removed from it
  {
    if(node->rtnode)
      fgDestroyRTNode(node->rtnode);
  }
  else if(!area)
  {
    parent = node->outline->parent;
    // On our initial layout, we have to calculate the extent by using only absolute coordinates and absolute centers
    node->extent = node->outline->area.abs;
    node->extent += node->outline->center.abs;

    // Then we calculate the entire proper position
    node->area = fgResolveArea(node->outline->area, node->outline->center, parent->node->area);

    // Now we create an R-tree node
    if(!node->rtnode)
    {
      node->rtnode = fgAllocRTNode();
      node->rtnode->document = node;
      node->rtnode->sibling = parent->node->rtnode->children;
      parent->node->rtnode->children = node->rtnode;
      node->rtnode->extent = node->extent;
      node->rtnode->area = node->area;
    }
    else if(node->rtnode->children)
    {
      fgMergeRect(&node->rtnode->area, &node->area);
      fgMergeRect(&node->rtnode->extent, &node->extent);
    }
    else
    {
      node->rtnode->area = node->area;
      node->rtnode->extent = node->extent;
    }

    fgFixRTNode(node->rtnode);
  }
  else // In this case, we are simply fixing relative coordinates
  {
    // Fix the area
    node->area = fgResolveArea(node->outline->area, node->outline->center, *area);

    if(node->rtnode->children)
      fgMergeRect(&node->rtnode->area, &node->area);
    else
      node->rtnode->area = node->area;

    // Now fix the R-tree node
    fgFixRTNode(node->rtnode);
  }
}

extern "C" unsigned int fgLayoutDefaultResolver(void* outline, unsigned int index, fgCalcNode* out, const char* id)
{
  //if(id)
  //  return ~0;

  return ~0;
}