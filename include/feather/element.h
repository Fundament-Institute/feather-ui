// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__ELEMENT_H
#define FG__ELEMENT_H

#include "message.h"
#include "khash.h"

KHASH_DECLARE(transition, int, unsigned long long);

struct FG__R_TREE_NODE;

// Represents the current base state of a layout node in the layout tree.
typedef struct FG__ELEMENT_NODE
{
  fgRect area; // This is relative to the parent R-Tree node, and is not a meaningful value
  fgRect layout;
  //fgRect clip, // Might not be necessary because it's equivilant to the RTreeNode's area
  float lineHeight;
  float fontSize;
  struct FG__ELEMENT_NODE* children;
  struct FG__ELEMENT_NODE* next;
  struct FG__ELEMENT_NODE* sibling; // Used only to track split flow nodes
  struct FG__R_TREE_NODE* node; // The corresponding RTreeNode 
  struct kh_transition_s* tstate; // timestamps mapped to a corresponding field for the last time it was changed
  fgBehaviorFunction behavior;
  fgBehaviorState* state;
} fgElementNode;

#endif