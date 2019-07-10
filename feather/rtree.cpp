// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/rtree.h"
#include "util.h"
#include <assert.h>
#include <algorithm>

extern "C" fgRect fgResolveNodeArea(fgRTNode* node)
{
  assert(node);
  fgRect r = node->area;

  node = node->parent;
  while(node)
  {
    r += node->area.topleft;
    node = node->parent;
  }

  return r;
}

extern "C" fgRTNode* fgAllocRTNode()
{
  return (fgRTNode*)calloc(1, sizeof(fgRTNode)); // TODO actually make a fixed-size allocator
}

extern "C" void fgDestroyRTNode(fgRTNode* node)
{
  if(node->parent)
  {
    if(node->parent->children == node)
      node->parent->children = node->sibling;
    else if(node->parent->children != nullptr)
    {
      for(fgRTNode* cur = node->parent->children; cur->sibling != nullptr; cur = cur->sibling)
        if(cur->sibling == node)
        {
          cur->sibling = node->sibling;
          break;
        }
    }
  }
  
  while(node->children) // This will efficiently remove the root child until there aren't any left
    fgDestroyRTNode(node->children);

  if(node->document)
    node->document->rtnode = 0;
  free(node);
}

extern "C" void fgFixRTNode(fgRTNode* node)
{
  bool fix = false;

  // Two fix cases: either we broke the topleft barrier, which requires recalculating everything, or we broke bottomright, which is easy to fix
  if(node->extent.left < 0 || node->extent.top < 0)
  {
    fgVec adjust = { std::max(-node->area.left, 0.0f), std::max(-node->area.top, 0.0f) };
    node->area += adjust;
    node->extent += adjust;
    node->parent->area -= adjust;
    node->parent->extent -= adjust;

    for(fgRTNode* n = node->children; n; n = n->sibling)
    {
      n->area += adjust; // We don't need to fix this because it couldn't have invalidated the child's area
      n->extent += adjust;
    }

    if(node->document)
    {
      node->document->area += adjust;
      node->document->extent += adjust;
    }

    fix = true;
  }

  if(node->extent.right > (node->parent->extent.right - node->parent->extent.left))
  {
    node->parent->extent.right = node->parent->extent.left + node->extent.right;
    fix = true;
  }
  if(node->extent.bottom > (node->parent->extent.bottom - node->parent->extent.top))
  {
    node->parent->extent.bottom = node->parent->extent.top + node->extent.bottom;
    fix = true;
  }
  if(node->area.right > (node->parent->area.right - node->parent->area.left))
  {
    node->parent->area.right = node->parent->area.left + node->area.right;
    fix = true;
  }
  if(node->area.bottom > (node->parent->area.bottom - node->parent->area.top))
  {
    node->parent->area.bottom = node->parent->area.top + node->area.bottom;
    fix = true;
  }

  if(fix)
    fgFixRTNode(node->parent);
}