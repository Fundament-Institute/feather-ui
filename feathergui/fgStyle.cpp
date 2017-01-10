// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "bss-util/bss_util.h"
#include "fgStyle.h"
#include "feathercpp.h"

KHASH_INIT(fgStyles, const char*, FG_UINT, 1, kh_str_hash_funcins, kh_str_hash_insequal);
FG_UINT fgStyleFlagMask = 0;

void fgStyle_Init(fgStyle* self)
{
  memset(self, 0, sizeof(fgStyle));
}

void fgStyle_Destroy(fgStyle* self)
{
  while(self->styles)
    fgStyle_RemoveStyleMsg(self, self->styles);
}

fgStyleMsg* fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2)
{
  fgStyleMsg* r = (fgStyleMsg*)fgmalloc<char>(sizeof(fgStyleMsg) + arglen1 + arglen2, __FILE__, __LINE__);
  memcpy(&r->msg, msg, sizeof(FG_Msg));

  if(arg1)
  {
    r->msg.p = r + 1;
    memcpy(r->msg.p, arg1, arglen1);
  }

  if(arg2)
  {
    r->msg.p2 = ((char*)(r + 1)) + arglen1;
    memcpy(r->msg.p2, arg2, arglen2);
  }

  r->next = self->styles;
  self->styles = r;
  return r;
}

void fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg)
{
  if(self->styles == msg)
    self->styles = msg->next;
  else
  {
    fgStyleMsg* cur = self->styles;
    while(cur && cur->next != msg) cur = cur->next;
    if(cur) cur->next = msg->next;
  }
  fgfree(msg, __FILE__, __LINE__);
}


fgStyleMsg* fgStyle::AddStyleMsg(const FG_Msg* msg) { return fgStyle_AddStyleMsg(this, msg, 0,0,0,0); }
void fgStyle::RemoveStyleMsg(fgStyleMsg* msg) { fgStyle_RemoveStyleMsg(this, msg); }

struct fgStyleStatic
{
  fgStyleStatic() { h = kh_init_fgStyles(); }
  ~fgStyleStatic() {
    for(khiter_t i = 0; i < h->n_buckets; ++i)
    {
      if(kh_exist(h, i))
        fgfree(const_cast<char*>(kh_key(h, i)), __FILE__, __LINE__);
    }
    kh_destroy_fgStyles(h);
  }
  kh_fgStyles_t* h;
};
FG_UINT fgStyle_GetName(const char* name, char flag)
{
  static fgStyleStatic stylehash;
  static FG_UINT count = 0;
  assert(count < (sizeof(FG_UINT)<<3));
  
  int r;
  khiter_t iter = kh_put_fgStyles(stylehash.h, name, &r);
  if(r) // if it wasn't in there before, we need to initialize the index
  {
    kh_key(stylehash.h, iter) = fgCopyText(name, __FILE__, __LINE__);
    kh_val(stylehash.h, iter) = (1 << count++);
    if(flag) fgStyleFlagMask |= kh_val(stylehash.h, iter);
  }
  assert(!!(fgStyleFlagMask & kh_val(stylehash.h, iter)) == flag);
  return kh_val(stylehash.h, iter);
}