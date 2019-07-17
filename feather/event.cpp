// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/event.h"
#include "feather/root.h"
#include "util.h"
#include <assert.h>
#include <utility>
#include <stdint.h>

inline khint_t event_tuple_hash(const fgEventTuple& t)
{
  khint_t i = kh_int64_hash_func(static_cast<uint64_t>(kh_ptr_hash_func(t.source)) | (static_cast<uint64_t>(kh_ptr_hash_func(t.dest)) << 32));
  khint_t field = kh_int64_hash_func(static_cast<uint64_t>(t.srcfield) | (static_cast<uint64_t>(t.destfield) << 32));
  return kh_int64_hash_func(static_cast<uint64_t>(i) | (static_cast<uint64_t>(field) << 32));
}

inline khint_t event_tuple_equal(const fgEventTuple& l, const fgEventTuple& r)
{
  return l.srcfield == r.srcfield && l.source == r.source && l.destfield == r.destfield && l.dest == r.dest;
}

__KHASH_IMPL(event, extern "C", unsigned int, struct FG__EVENT*, 1, kh_int_hash_func, kh_int_hash_equal);
__KHASH_IMPL(eventnode, extern "C", fgEventTuple, unsigned int, 1, event_tuple_hash, event_tuple_equal);

extern "C" fgEvent* fgAllocEvent(void* self, fgListener fn, unsigned int srcfield, kh_event_t* source, unsigned int destfield, struct FG__EVENT** dest, unsigned int aux)
{
  auto e = (fgEvent*)calloc(1, sizeof(fgEvent) + aux);
  if(!e)
    return 0;
  *e = { (!self && aux > 0) ? (e + 1) : self, fn, source, dest, srcfield, destfield };

  if(e->source)
  {
    int r;
    khiter_t iter = kh_put_event(e->source, e->srcfield, &r);
    if(r >= 0)
    {
      if(r > 0)
        kh_value(e->source, iter) = 0;

      LLAdd(e, kh_value(e->source, iter));
    }
  }
  if(e->dest)
  {
    e->objprev = 0;
    e->objnext = *e->dest;
    if(*e->dest) (*e->dest)->objprev = e;
    *e->dest = e;
  }

  return e;
}

extern "C" void fgDestroyEvent(struct FG__ROOT* root, fgEvent* e)
{
  fgRemoveEvent(root, e);
  free(e);
}

extern "C" void fgRemoveEvent(struct FG__ROOT* root, fgEvent* e)
{
  if(e->source || e->dest)
  {
    khiter_t iter = kh_get_eventnode(root->eventnodes, fgEventTuple{ e->source, e->dest, e->srcfield, e->destfield });
    if(iter < kh_end(root->eventnodes) && kh_exist(root->eventnodes, iter))
    {
      if(--kh_val(root->eventnodes, iter) > 0)
        return;
      kh_del_eventnode(root->eventnodes, iter);
    }
  }
  if(e->source)
  {
    khiter_t iter = kh_get_event(e->source, e->srcfield);
    if(iter < kh_end(e->source) && kh_exist(e->source, iter))
    {
      LLRemove(e, kh_value(e->source, iter));
      if(!kh_value(e->source, iter))
        kh_del_event(e->source, iter);
    }
    e->source = 0;
  }
  if(e->dest)
  {
    if(e->objprev != 0) e->objprev->objnext = e->objnext;
    else if(*e->dest == e)* e->dest = e->objnext;
    if(e->objnext != 0) e->objnext->objprev = e->objprev;
    e->dest = 0;
  }
}

extern "C" void fgSendEvent(struct FG__ROOT* root, kh_event_t* source, unsigned int key, void* target, void* field)
{
  khiter_t i = kh_get_event(source, key);
  if(i < kh_end(source) && kh_exist(source, i))
  {
    auto cur = kh_val(source, i);
    while(cur)
    {
      (*cur->event)(root, cur->self, target, cur->destfield, field);
      cur = cur->next;
    }
  }
}