// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/outline.h"
#include "feather/rtree.h"
#include "util.h"
#include <assert.h>

extern "C" {
  __KHASH_IMPL(transition, , int, unsigned long long, 1, kh_int_hash_func, kh_int_hash_equal);
}

extern "C" fgDocumentNode* fgAllocDocumentNode()
{
  return (fgDocumentNode*)calloc(1, sizeof(fgDocumentNode)); // TODO: fixed arena allocator
}

extern "C" void fgDestroyDocumentNode(struct FG__ROOT* root, fgDocumentNode* node, float scale, fgVec dpi)
{
  // Remove all listeners
  fgEventRef* cur;
  fgEventRef* next = node->events;
  while(cur = next)
  {
    auto iter = kh_get_event(cur->node->listeners, cur->field);
    if(iter != kh_end(cur->node->listeners) && kh_exist(cur->node->listeners, iter))
      LLRemove<fgEvent>(cur->ptr, kh_val(cur->node->listeners, iter));
    fgDestroyEvent(cur->ptr);
    next = cur;
    fgDestroyEventRef(cur);
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


extern "C" fgEvent* fgAllocEvent(fgDocumentNode* obj, fgListener listener)
{
  auto e = (fgEvent*)calloc(1, sizeof(fgEvent)); // TODO: fixed arena allocator
  if(e)
  {
    e->event = listener;
    e->object = obj;
  }
  return e;
}
extern "C" void fgDestroyEvent(fgEvent* e)
{
  free(e);
}
extern "C" void fgRemoveEvent(fgEvent* e)
{
  if(e->object->events->ptr == e)
    e->object->events = e->object->events->next;

  for(fgEventRef* cur = e->object->events; cur->next; cur = cur->next)
  {
    if(cur->next->ptr == e)
    {
      fgEventRef* next = cur->next->next;
      fgDestroyEventRef(cur->next);
      cur->next = next;
      return;
    }
  }
}

extern "C" fgEventRef* fgAllocEventRef()
{
  return (fgEventRef*)calloc(1, sizeof(fgEventRef)); // TODO: fixed arena allocator
}
extern "C" void fgDestroyEventRef(fgEventRef* e)
{
  free(e);
}