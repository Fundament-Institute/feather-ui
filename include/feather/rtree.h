// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__R_TREE_H
#define FG__R_TREE_H

#include "element.h"

typedef struct FG__R_TREE_NODE
{
  fgRect area;
  fgElementNode* node; // If this R-tree node has an attached element, this points to it
  struct FG__R_TREE_NODE* sibling;
  struct FG__R_TREE_NODE* children;
} fgRTreeNode;

#endif
