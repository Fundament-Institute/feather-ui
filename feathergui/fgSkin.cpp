// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "fgSkin.h"
#include "fgResource.h"
#include "fgText.h"
#include "feathercpp.h"

KHASH_INIT(fgSkins, const char*, fgSkin*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgStyles, const char*, FG_UINT, 1, kh_str_hash_funcins, kh_str_hash_insequal);

static_assert(sizeof(fgStyleLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgSubskinArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgStyleArray) == sizeof(fgVector), "mismatch between vector sizes");
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
  reinterpret_cast<fgStyleLayoutArray&>(self->children).~cDynArray();
  reinterpret_cast<fgSubskinArray&>(self->subskins).~cDynArray();
  reinterpret_cast<fgStyleArray&>(self->styles).~cDynArray();
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
size_t FG_FASTCALL fgSkin_AddChild(fgSkin* self, const char* name, const fgElement* element, fgFlag flags)
{
  return ((fgStyleLayoutArray&)self->children).AddConstruct(name, element, flags);
}
char FG_FASTCALL fgSkin_RemoveChild(fgSkin* self, FG_UINT child)
{
  return DynArrayRemove((fgFontArray&)self->children, child);
}
fgStyleLayout* FG_FASTCALL fgSkin_GetChild(const fgSkin* self, FG_UINT child)
{
  return self->children.p + child;
}

size_t FG_FASTCALL fgSkin_AddStyle(fgSkin* self, const char* name)
{
  FG_UINT r = fgStyle_GetName(name);
  if(r >= ((fgStyleArray&)self->styles).Length())
    ((fgStyleArray&)self->styles).SetLength(r + 1);
  return r;
}
char FG_FASTCALL fgSkin_RemoveStyle(fgSkin* self, FG_UINT style)
{
  return DynArrayRemove((fgStyleArray&)self->styles, style);
}
fgStyle* FG_FASTCALL fgSkin_GetStyle(const fgSkin* self, FG_UINT style)
{
  return ((fgStyleArray&)self->styles).begin() + style;
}

size_t FG_FASTCALL fgSkin_AddSubSkin(fgSkin* self, int index)
{
  return ((fgSubskinArray&)self->subskins).AddConstruct(index);
}

char FG_FASTCALL fgSkin_RemoveSubSkin(fgSkin* self, FG_UINT subskin)
{
  return DynArrayRemove((fgSubskinArray&)self->subskins, subskin);
}
fgSkin* FG_FASTCALL fgSkin_GetSubSkin(const fgSkin* self, FG_UINT subskin)
{
  return self->subskins.p + subskin;
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
    kh_val(self->skinmap, iter) = (fgSkin*)malloc(sizeof(fgSkin));
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
  if(!self->skinmap || !name)
    return 0;
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? kh_val(self->skinmap, iter) : 0;
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* name, const fgElement* element, fgFlag flags)
{
  self->name = fgCopyText(name);
  self->element = *element;
  self->flags = flags;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  if(self->name) free(self->name);
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

  int r;
  khiter_t iter = kh_put_fgStyles(h, name, &r);
  if(r) // if it wasn't in there before, we need to initialize the index
    kh_val(h, iter) = count++;
  return kh_val(h, iter);
}