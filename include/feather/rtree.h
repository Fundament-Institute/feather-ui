// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__R_TREE_H
#define FG__R_TREE_H

#include "document.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct FG__R_TREE_NODE
{
  fgRect area; // This is relative to the top-left corner of the parent area. Because the parent most encompass all children, it can't be negative.
  fgRect extent; // This is the minimal bounding rectangle of it's children, which respects clipping
  int top_zindex; // R-tree nodes are actually 3 dimensional, but the z-axis can never overlap (because layout rects have no depth).
  int bottom_zindex; // We use this property to keep all children sorted from highest zindex to lowest.
  float* transform; // Exposes the 4x4 affine transform that should be applied to this node (TODO: actually use this)
  fgDocumentNode* document; // If this R-tree node has an attached document state, this points to it
  struct FG__R_TREE_NODE* parent;
  struct FG__R_TREE_NODE* sibling;
  struct FG__R_TREE_NODE* children;
} fgRTNode;

FG_COMPILER_DLLEXPORT fgRect fgResolveNodeArea(fgRTNode* node); // Traverses up the R-tree to get the true area of a given node. Only used when a proper traversal is impossible.
FG_COMPILER_DLLEXPORT fgRTNode* fgAllocRTNode(); // Allocates a new node from an efficient, internal fixed-block allocator
FG_COMPILER_DLLEXPORT void fgDestroyRTNode(fgRTNode* node);
FG_COMPILER_DLLEXPORT void fgFixRTNode(fgRTNode* node); // After a node has been modified, it could potentially be outside the bounds of it's parent. This recursively fixes the R-tree.

#ifdef  __cplusplus
}
#endif

#endif
