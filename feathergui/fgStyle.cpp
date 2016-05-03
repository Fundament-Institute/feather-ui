// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "bss-util/bss_util.h"
#include "fgStyle.h"

KHASH_INIT(fgStyles, const char*, FG_UINT, 1, kh_str_hash_funcins, kh_str_hash_insequal);

void FG_FASTCALL fgStyle_Init(fgStyle* self)
{
  memset(self, 0, sizeof(fgStyle));
}

void FG_FASTCALL fgStyle_Destroy(fgStyle* self)
{
  while(self->styles)
    fgStyle_RemoveStyleMsg(self, self->styles);
}

fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2)
{
  fgStyleMsg* r = (fgStyleMsg*)malloc(sizeof(fgStyleMsg) + arglen1 + arglen2);
  memcpy(&r->msg, msg, sizeof(FG_Msg));

  if(arg1)
  {
    r->msg.other = r + 1;
    memcpy(r->msg.other, arg1, arglen1);
  }

  if(arg2)
  {
    r->msg.other2 = ((char*)(r + 1)) + arglen1;
    memcpy(r->msg.other2, arg2, arglen2);
  }

  r->next = self->styles;
  self->styles = r;
  return r;
}

void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg)
{
  if(self->styles == msg)
    self->styles = msg->next;
  else
  {
    fgStyleMsg* cur = self->styles;
    while(cur && cur->next != msg) cur = cur->next;
    if(cur) cur->next = msg->next;
  }
  free(msg);
}

FG_UINT FG_FASTCALL fgStyle_GetName(const char* name)
{
  static kh_fgStyles_t* h = kh_init_fgStyles();
  static FG_UINT count = 0;
  assert(count < (sizeof(FG_UINT)<<3));

  int r;
  khiter_t iter = kh_put_fgStyles(h, name, &r);
  if(r) // if it wasn't in there before, we need to initialize the index
    kh_val(h, iter) = (1 << count++);
  return kh_val(h, iter);
}