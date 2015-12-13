// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "fgSkin.h"
#include "fgResource.h"
#include "fgText.h"
#include "feathercpp.h"

KHASH_INIT(fgSkins, const char*, fgSkin, 1, kh_str_hash_funcins, kh_str_hash_insequal);

static_assert(sizeof(fgStyleLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgStyleArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgSubskinArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgClassLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");

void FG_FASTCALL fgSkin_Init(fgSkin* self)
{
  memset(self, 0, sizeof(fgSkin));
}

void FG_FASTCALL fgSubskin_Init(fgSkin* self, int index)
{
  memset(self, 0, sizeof(fgSkin));
  self->index = index;
}

char FG_FASTCALL fgSkin_DestroySkinElement(fgSkin* self, khiter_t iter)
{
  if(kh_exist(self->skinmap, iter))
  {
    fgSkin_Destroy(&kh_val(self->skinmap, iter));
    free((char*)kh_key(self->skinmap, iter));
    kh_del(fgSkins, self->skinmap, iter);
    return 1;
  }
  return 0;
}

void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  fgStyle_Destroy(&self->style);
  ((fgResourceArray&)self->resources).~cDynArray();
  ((fgFontArray&)self->fonts).~cDynArray();
  ((fgStyleLayoutArray&)self->children).~cDynArray();
  ((fgStyleArray&)self->styles).~cDynArray();
  ((fgSubskinArray&)self->subskins).~cDynArray();
  
  if(self->skinmap)
  {
    khiter_t cur = kh_begin(self->skinmap);
    while(cur != kh_end(self->skinmap)) fgSkin_DestroySkinElement(self, cur++);
    kh_destroy_fgSkins(self->skinmap);
  }
}
FG_UINT FG_FASTCALL fgSkin_AddResource(fgSkin* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgSkin_RemoveResource(fgSkin* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgSkin_GetResource(const fgSkin* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
FG_UINT FG_FASTCALL fgSkin_AddFont(fgSkin* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgSkin_RemoveFont(fgSkin* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgSkin_GetFont(const fgSkin* self, FG_UINT font)
{
  return DynGet<fgFontArray>(self->fonts, font).ref;
}
FG_UINT FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* name, fgElement* element, fgFlag flags)
{
  return ((fgStyleLayoutArray&)self->children).AddConstruct(name, element, flags);
}
char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child)
{
  return DynGetP<fgStyleLayoutArray>(self->children, child);
}

FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self)
{
  return ((fgStyleArray&)self->styles).AddConstruct();
}
char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style)
{
  return DynArrayRemove((fgStyleArray&)self->styles, style);
}
fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style)
{
  return ((fgStyleArray&)self->styles).begin() + style;
}

FG_UINT FG_FASTCALL fgSkin_AddSubSkin(fgSkin* self, int index)
{
  return ((fgSubskinArray&)self->subskins).AddConstruct(index);
}

char FG_FASTCALL fgSkin_RemoveSubSkin(fgSkin* self, FG_UINT subskin)
{
  return DynArrayRemove((fgSubskinArray&)self->subskins, subskin);
}
fgSkin* FG_FASTCALL fgSkin_GetSubSkin(const fgSkin* self, FG_UINT subskin)
{
  return DynGetP<fgSubskinArray>(self->subskins, subskin);
}

fgSkin* FG_FASTCALL fgSkin_AddSkin(fgSkin* self, const char* name)
{
  if(!self->skinmap)
    self->skinmap = kh_init_fgSkins();

  if(!name) return 0;
  int r = 0;
  khiter_t iter = kh_put(fgSkins, self->skinmap, name, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    fgSkin_Init(&kh_val(self->skinmap, iter));
    kh_key(self->skinmap, iter) = fgCopyText(name);
  }

  return &kh_val(self->skinmap, iter);
}
char FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, const char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return fgSkin_DestroySkinElement(self, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name)
{
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? (&kh_val(self->skinmap, iter)) : 0;
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* name, fgElement* element, fgFlag flags)
{
  self->name = fgCopyText(name);
  self->element = *element;
  self->flags = flags;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  free(self->name);
  fgStyle_Destroy(&self->style);
}

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
    r->msg.other1 = r + 1;
    memcpy(r->msg.other1, arg1, arglen1);
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