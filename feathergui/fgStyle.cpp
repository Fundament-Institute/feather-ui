// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "bss-util/khash.h"
#include "bss-util/bss_util.h"
#include "fgStyle.h"

KHASH_INIT(fgStyles, const char*, FG_UINT, 1, kh_str_hash_funcins, kh_str_hash_insequal);
fgStyleStatic fgStyleStatic::Instance;

fgStyleStatic::fgStyleStatic() : h(0) {}
fgStyleStatic::~fgStyleStatic() { Clear(); }
void fgStyleStatic::Init()
{
  Clear();
  h = kh_init_fgStyles();
  bss::bssFill(Masks, 0);
  for(size_t i = 0; i < (sizeof(fgStyleIndex) << 3); ++i)
    Masks[i] = (1 << i);
}
void fgStyleStatic::Clear()
{
  if(h)
  {
    for(khiter_t i = 0; i < h->n_buckets; ++i)
    {
      if(kh_exist(h, i))
        fgFreeText(kh_key(h, i), __FILE__, __LINE__);
    }
    kh_destroy_fgStyles(h);
    h = 0;
  }
}

void fgStyle_Init(fgStyle* self)
{
  bss::bssFill(*self, 0);
}

void fgStyle_InitCopy(fgStyle* self, const fgStyle* from)
{
  self->styles = 0;
  for(fgStyleMsg* cur = from->styles; cur != 0; cur = cur->next)
  {
    fgStyleMsg* m = fgStyle_CloneStyleMsg(cur);
    m->next = self->styles;
    self->styles = m;
  }
}

void fgStyle_Destroy(fgStyle* self)
{
  while(self->styles)
    fgStyle_RemoveStyleMsg(self, self->styles);
}

fgStyleMsg* fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, unsigned int arg1size, unsigned int arg2size)
{
  unsigned int sz = sizeof(fgStyleMsg) + arg1size + arg2size;
  assert(!(sz & 0xC0000000));
  fgStyleMsg* r = (fgStyleMsg*)fgmalloc<char>(sz, __FILE__, __LINE__);
  MEMCPY(&r->msg, sizeof(FG_Msg), msg, sizeof(FG_Msg));

  if(arg1size > 0)
  {
    void* arg1 = r->msg.p;
    r->msg.p = r + 1;
    memcpy(r->msg.p, arg1, arg1size);
    sz |= 0x40000000;
  }

  if(arg2size > 0)
  {
    void* arg2 = r->msg.p2;
    r->msg.p2 = ((char*)(r + 1)) + arg1size;
    memcpy(r->msg.p2, arg2, arg2size);
    sz |= 0x80000000;
  }

  r->sz = sz;
  r->next = self->styles;
  self->styles = r;
  return r;
}

fgStyleMsg* fgStyle_CloneStyleMsg(const fgStyleMsg* self)
{
  unsigned int sz = self->sz&(~0xC0000000);
  fgStyleMsg* r = (fgStyleMsg*)fgmalloc<char>(sz, __FILE__, __LINE__);
  MEMCPY(r, sz, self, sz);

  if(r->sz & 0x40000000)
    r->msg.p = (char*)r + ((char*)self->msg.p - (char*)self);

  if(r->sz & 0x80000000)
    r->msg.p2 = (char*)r + ((char*)self->msg.p2 - (char*)self);

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


fgStyleMsg* fgStyle::AddStyleMsg(const FG_Msg* msg, unsigned int arg1size, unsigned int arg2size) { return fgStyle_AddStyleMsg(this, msg, arg1size, arg2size); }
void fgStyle::RemoveStyleMsg(fgStyleMsg* msg) { fgStyle_RemoveStyleMsg(this, msg); }

FG_UINT fgStyle_GetName(const char* name)
{
  static FG_UINT count = 0;
  assert(count < (sizeof(FG_UINT) << 3));

  int r;
  khiter_t iter = kh_put_fgStyles(fgStyleStatic::Instance.h, name, &r);
  if(r) // if it wasn't in there before, we need to initialize the index
  {
    kh_key(fgStyleStatic::Instance.h, iter) = fgCopyText(name, __FILE__, __LINE__);
    kh_val(fgStyleStatic::Instance.h, iter) = (1 << count++);
  }
  return kh_val(fgStyleStatic::Instance.h, iter);
}

fgStyleIndex fgStyle_GetIndexGroups(fgStyleIndex index)
{
  fgStyleIndex mask = 0;
  while(index)
  {
    fgStyleIndex indice = bss::bssLog2(index);
    mask |= fgStyleStatic::Instance.Masks[indice];
    index ^= (1 << indice);
  }
  return mask;
}
fgStyleIndex fgStyle_AddGroup(fgStyleIndex group)
{
  if(!group)
    return 0;
  fgStyleIndex index = group;
  while(index)
  {
    fgStyleIndex indice = bss::bssLog2(index);
    if(fgStyleStatic::Instance.Masks[indice] != (1 << indice))
      return fgStyleStatic::Instance.Masks[indice];
    index ^= (1 << indice);
  }

  index = group;
  while(index)
  {
    fgStyleIndex indice = bss::bssLog2(index);
    fgStyleStatic::Instance.Masks[indice] = group;
    index ^= (1 << indice);
  }
  return 0;
}
fgStyleIndex fgStyle_AddGroupNames(size_t n, ...)
{
  fgStyleIndex group = 0;
  va_list vl;
  va_start(vl, n);
  for(size_t i = 0; i < n; i++)
    group |= fgStyle_GetName(va_arg(vl, const char*));
  va_end(vl);
  return fgStyle_AddGroup(group);
}

const char* fgStyle_GetMapIndex(fgStyleIndex index)
{
  for(khiter_t i = 0; i < kh_end(fgStyleStatic::Instance.h); ++i)
    if(kh_exist(fgStyleStatic::Instance.h, i) && kh_val(fgStyleStatic::Instance.h, i) == index)
      return kh_key(fgStyleStatic::Instance.h, i);
  return 0;
}

fgStyleIndex fgStyle_GetAllNames(const char* names)
{
  size_t len = strlen(names) + 1;
  VARARRAY(char, tokenize, len);
  MEMCPY(tokenize, len, names, len);
  char* context;
  char* token = STRTOK(tokenize, "+", &context);
  fgStyleIndex style = 0;
  while(token)
  {
    style |= fgStyle_GetName(token);
    token = STRTOK(0, "+", &context);
  }

  return style;
}