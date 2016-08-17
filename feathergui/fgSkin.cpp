// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "fgSkin.h"
#include "fgResource.h"
#include "fgText.h"
#include "feathercpp.h"

KHASH_INIT(fgSkins, const char*, fgSkin*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgStyleInt, FG_UINT, fgStyle, 1, kh_int_hash_func, kh_int_hash_equal);

static_assert(sizeof(fgStyleLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgStyleArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgClassLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");

void FG_FASTCALL fgSkin_Init(fgSkin* self)
{
  memset(self, 0, sizeof(fgSkin));
}

template<class HASH, class T, void (FG_FASTCALL *DESTROY)(T*), void(*DEL)(HASH*, khint_t)>
char FG_FASTCALL DestroyHashElement(HASH* self, khiter_t iter)
{
  if(kh_exist(self, iter))
  {
    (*DESTROY)(kh_val(self, iter));
    free(kh_val(self, iter));
    free((char*)kh_key(self, iter));
    (*DEL)(self, iter);
    return 1;
  }
  return 0;
}

template<class HASH, class T, void(FG_FASTCALL *DESTROYELEM)(T*), void(*DEL)(HASH*, khint_t), void(*DESTROYHASH)(HASH*)>
void FG_FASTCALL DestroyHash(HASH* self)
{
  if(self)
  {
    khiter_t cur = kh_begin(self);
    while(cur != kh_end(self)) DestroyHashElement<HASH, T, DESTROYELEM, DEL>(self, cur++);
    (*DESTROYHASH)(self);
  }
}

void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  fgStyle_Destroy(&self->style);
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  reinterpret_cast<fgStyleLayoutArray&>(self->children).~cArraySort();

  if(self->styles != 0)
  {
    khiter_t cur = kh_begin(self->styles);
    while(cur != kh_end(self->styles))
    {
      if(kh_exist(self->styles, cur))
        fgStyle_Destroy(self->styles->vals + cur);
      ++cur;
    }
    kh_destroy_fgStyleInt(self->styles);
  }

  DestroyHash<kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins, &kh_destroy_fgSkins>(self->skinmap);
}
size_t FG_FASTCALL fgSkin_AddResource(fgSkin* self, void* resource)
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
size_t FG_FASTCALL fgSkin_AddFont(fgSkin* self, void* font)
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
size_t FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  return ((fgStyleLayoutArray&)self->children).Insert(fgStyleLayoutConstruct(type, name, flags, transform, order));
}
char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child)
{
  return self->children.p + child;
}

FG_UINT FG_FASTCALL fgSkin_AddStyle(fgSkin* self, const char* name)
{
  if(!self->styles)
    self->styles = kh_init_fgStyleInt();

  size_t len = strlen(name) + 1;
  DYNARRAY(char, tokenize, len);
  MEMCPY(tokenize, len, name, len);
  char* context;
  char* token = STRTOK(tokenize, "+", &context);
  FG_UINT style = 0;
  while(token)
  {
    style |= fgStyle_GetName(token, style != 0); // If this is the first token we're parsing, it's not a flag, otherwise it is a flag.
    token = STRTOK(0, "+", &context);
  }

  int r = 0;
  khiter_t i = kh_put_fgStyleInt(self->styles, style, &r);

  if(r != 0)
    kh_val(self->styles, i).styles = 0;
  return style;
}
char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style)
{
  khiter_t i = kh_get_fgStyleInt(self->styles, style);
  if(i < kh_end(self->styles) && kh_exist(self->styles, i))
  {
    fgStyle_Destroy(self->styles->vals + i);
    kh_del_fgStyleInt(self->styles, i);
    return 1;
  }
  return 0;
}
fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style)
{
  if(!self->styles) return 0;
  khiter_t i = kh_get_fgStyleInt(self->styles, style);
  return (i < kh_end(self->styles) && kh_exist(self->styles, i)) ? (self->styles->vals + i) : 0;
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
    kh_val(self->skinmap, iter) = bss_util::bssmalloc<fgSkin>(1);
    fgSkin_Init(kh_val(self->skinmap, iter));
    kh_key(self->skinmap, iter) = fgCopyText(name);
  }

  return kh_val(self->skinmap, iter);
}
char FG_FASTCALL fgSkin_RemoveSkin(fgSkin* self, const char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return DestroyHashElement<struct __kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins>(self->skinmap, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name)
{
  if(!self || !name)
    return 0;
  if(!self->skinmap)
	return fgSkin_GetSkin(self->inherit, name);
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? kh_val(self->skinmap, iter) : fgSkin_GetSkin(self->inherit, name);
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  self->type = fgCopyText(type);
  self->name = fgCopyText(name);
  self->transform = *transform;
  self->flags = flags;
  self->order = order;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  if(self->type) free(self->type);
  if(self->name) free(self->name);
  fgStyle_Destroy(&self->style);
}