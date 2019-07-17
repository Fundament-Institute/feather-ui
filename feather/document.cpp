// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/outline.h"
#include "feather/rtree.h"
#include "util.h"
#include <assert.h>

__KHASH_IMPL(transition, extern "C", int, unsigned long long, 1, kh_int_hash_func, kh_int_hash_equal);

extern "C" fgDocumentNode* fgAllocDocumentNode()
{
  return (fgDocumentNode*)calloc(1, sizeof(fgDocumentNode)); // TODO: fixed arena allocator
}

extern "C" void fgDestroyDocumentNode(struct FG__ROOT* root, fgDocumentNode* node, float scale, fgVec dpi)
{
  // Remove all listeners
  fgEvent* cur;
  fgEvent* next = node->events;
  while(cur = next)
  {
    next = cur->next;
    fgDestroyEvent(root, cur);
  }

  if(node->tstate)
    kh_destroy_transition(node->tstate);

  // Remove from parent, which will alert the layout node and remove us from the R-tree
  if(node->outline && node->outline->parent)
  {
    fgMessage msg = { FG_MSG_REMOVE_CHILD };
    msg.child.dpi = dpi;
    msg.child.scale = scale;
    msg.child.node = node;
    fgSendMessage(root, node->outline->parent->node, &msg);
    assert(!node->rtnode);
  }
  else
    fgDestroyRTNode(node->rtnode);

  // Remove from outline node
  if(node->outline)
    node->outline->node = 0;
  free(node);
}