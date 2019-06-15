// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__OUTLINE_NODE_H
#define FG__OUTLINE_NODE_H

#include "element.h"

struct FG__OUTLINE_NODE;
typedef struct {} fgLayoutData;
typedef void (*fgGenerator)(struct FG__OUTLINE_NODE*);
typedef void (*fgLayout)(struct FG__OUTLINE_NODE*, fgElementNode*);

// The outline describes all the metadata required to construct the layout element state.
typedef struct FG__OUTLINE_NODE
{
  URect area;
  UVec center;
  fgRect margin;
  fgRect padding;
  fgVec min;
  fgVec max;
  float lineHeight;
  float fontSize;
  fgFlag layoutFlags;
  fgFlag stateFlags;
  FG__OUTLINE_NODE* children;
  FG__OUTLINE_NODE* sibling;
  fgGenerator generator;
  fgElementNode* node; // instantiated layout element state node
  fgBehaviorFunction behavior;
  fgLayout display; // Despite being passed the LayoutNode, the layout function actually operates on the RTreeNode subtree owned by this outline node.
  fgLayoutData* data; // layout function specific data (like "flex-justify") must be stored after any behavior parameter extensions (like fgLayoutLayerNode). This points to where it is located. The generator function is allowed to change this data, because it might be bound to stateful layout information.
} fgOutlineNode;

enum LAYER_MOUSE
{
  LAYER_MOUSE_LAYOUT = 1,
  LAYER_MOUSE_TRANSFORM = 2,
  LAYER_MOUSE_BOTH = 3,
};

// This is an extension of fgLayoutNode that contains additional information for a "layer"
typedef struct
{
  fgOutlineNode layout;
  float transform[4][4];
  float opacity;
  enum LAYER_MOUSE mouse;
} fgOutlineLayerNode;

#endif
