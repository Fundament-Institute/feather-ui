// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "fgSkin.h"
#include "fgResource.h"
#include "fgText.h"
#include "feathercpp.h"
#include "bss-util/cXML.h"

#include <fstream>
#include <sstream>

KHASH_INIT(fgSkins, const char*, fgSkin*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgStyleInt, FG_UINT, fgStyle, 1, kh_int_hash_func, kh_int_hash_equal);

static_assert(sizeof(fgStyleLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgStyleArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgClassLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");

using namespace bss_util;

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
void FG_FASTCALL fgSkinBase_Destroy(fgSkinBase* self)
{
  reinterpret_cast<fgResourceArray&>(self->resources).~cDynArray();
  reinterpret_cast<fgFontArray&>(self->fonts).~cDynArray();
  DestroyHash<kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins, &kh_destroy_fgSkins>(self->skinmap);
}

void FG_FASTCALL fgSkin_Destroy(fgSkin* self)
{
  fgStyle_Destroy(&self->style);
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

  fgSkinBase_Destroy(&self->base);
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

fgSkin* FG_FASTCALL fgSkin_GetSkin(const fgSkin* self, const char* name)
{
  if(!self || !name)
    return 0;
  fgSkin* r = fgSkinBase_GetSkin(&self->base, name);
  return !r ? fgSkin_GetSkin(self->inherit, name) : r;
}

size_t FG_FASTCALL fgSkinBase_AddResource(fgSkinBase* self, void* resource)
{
  return ((fgResourceArray&)self->resources).Add(resource);
}
char FG_FASTCALL fgSkinBase_RemoveResource(fgSkinBase* self, FG_UINT resource)
{
  return DynArrayRemove((fgResourceArray&)self->resources, resource);
}
void* FG_FASTCALL fgSkinBase_GetResource(const fgSkinBase* self, FG_UINT resource)
{
  return DynGet<fgResourceArray>(self->resources, resource).ref;
}
size_t FG_FASTCALL fgSkinBase_AddFont(fgSkinBase* self, void* font)
{
  return ((fgFontArray&)self->fonts).Add(font);
}
char FG_FASTCALL fgSkinBase_RemoveFont(fgSkinBase* self, FG_UINT font)
{
  return DynArrayRemove((fgFontArray&)self->fonts, font);
}
void* FG_FASTCALL fgSkinBase_GetFont(const fgSkinBase* self, FG_UINT font)
{
  return DynGet<fgFontArray>(self->fonts, font).ref;
}
fgSkin* FG_FASTCALL fgSkinBase_AddSkin(fgSkinBase* self, const char* name)
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
char FG_FASTCALL fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return DestroyHashElement<struct __kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins>(self->skinmap, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* FG_FASTCALL fgSkinBase_GetSkin(const fgSkinBase* self, const char* name)
{
  if(!self || !name || !self->skinmap)
    return 0;
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? kh_val(self->skinmap, iter) : 0;
}

void FG_FASTCALL fgStyleLayout_Init(fgStyleLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, int order)
{
  self->type = fgCopyText(type);
  self->name = fgCopyText(name);
  self->id = 0;
  self->transform = *transform;
  self->flags = flags;
  self->order = order;
  fgStyle_Init(&self->style);
}
void FG_FASTCALL fgStyleLayout_Destroy(fgStyleLayout* self)
{
  if(self->type) free(self->type);
  if(self->name) free(self->name);
  if(self->id) free(self->id);
  fgStyle_Destroy(&self->style);
}

void FG_FASTCALL fgSkins_LoadSubNodeXML(fgSkin* self, const cXMLNode* node)
{

}

fgSkin* FG_FASTCALL fgSkins_LoadNodeXML(struct __kh_fgSkins_t* self, const cXMLNode* root)
{
  const char* id = root->GetAttributeString("id");
  if(!id) return 0;
  int r = 0;
  khiter_t iter = kh_put_fgSkins(self, id, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    kh_val(self, iter) = bss_util::bssmalloc<fgSkin>(1);
    fgSkin_Init(kh_val(self, iter));
    kh_key(self, iter) = fgCopyText(id);
  }

  fgSkins_LoadSubNodeXML(kh_val(self, iter), root);
  return kh_val(self, iter);
}

fgSkin* FG_FASTCALL fgSkins_LoadStreamXML(struct __kh_fgSkins_t* self, std::istream& s)
{
  cXML xml(s);
  size_t index = 0;
  const cXMLNode* root;
  fgSkin* ret = 0;

  while(root = xml.GetNode(index++))
  {
    if(!stricmp(root->GetName(), "fg:skin"))
      ret = fgSkins_LoadNodeXML(self, root);
  }

  return ret;
}

fgSkin* FG_FASTCALL fgSkins_LoadFileXML(struct __kh_fgSkins_t* self, const char* file)
{
  return fgSkins_LoadStreamXML(self, std::ifstream(file, std::ios_base::in));
}
fgSkin* FG_FASTCALL fgSkins_LoadXML(struct __kh_fgSkins_t* self, const char* data, FG_UINT length)
{
  return fgSkins_LoadStreamXML(self, std::stringstream(std::string(data, length)));
}